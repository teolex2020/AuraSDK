//! Phase 4 Evidence Collection: Limited Influence Activation for Candidate B.
//!
//! Dual-mode comparison: baseline (Off) vs limited rerank for every query.
//! Collects per-query metrics, heuristic quality labels, contradiction delta,
//! aggregate statistics, and renders a decision against defined thresholds.
//!
//! This is the evidence artifact for the Phase 4 activation decision.

use aura::{Aura, Level};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

fn store_batch(aura: &Aura, batch: &[(&str, Level, &[&str], &str, &str)]) {
    for (content, level, tags, source, semantic) in batch {
        aura.store(
            content,
            Some(*level),
            Some(tags.iter().map(|t| t.to_string()).collect()),
            None, None,
            Some(*source),
            None,
            Some(false),
            None, None,
            Some(*semantic),
        )
        .unwrap_or_else(|e| panic!("store failed for '{}': {}", content, e));
    }
}

fn run_cycles(aura: &Aura, n: usize) {
    for _ in 0..n {
        aura.run_maintenance();
    }
}

// ─────────────────────────────────────────────────────────
// Per-query evidence
// ─────────────────────────────────────────────────────────

/// Heuristic quality label.
#[derive(Debug, Clone, Copy, PartialEq)]
enum QualityLabel {
    Better,
    Same,
    Worse,
    Unclear,
}

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

/// Per-query evidence record.
struct QueryEvidence {
    query: String,
    category: String,
    top_k: usize,
    baseline_count: usize,
    limited_count: usize,
    baseline_ids: Vec<String>,
    limited_ids: Vec<String>,
    top_k_overlap: f32,
    records_moved: usize,
    max_up_shift: usize,
    max_down_shift: usize,
    belief_coverage: f32,
    avg_belief_multiplier: f32,
    baseline_latency_us: u64,
    limited_latency_us: u64,
    latency_delta_us: i64,
    rerank_was_applied: bool,
    skip_reason: String,
    // Contradiction tracking: count of unresolved-belief records in top-k
    baseline_unresolved_in_top_k: usize,
    limited_unresolved_in_top_k: usize,
    contradiction_delta: i32, // negative = fewer unresolved in top-k = good
    label: QualityLabel,
}

/// Aggregate decision metrics.
struct AggregateDecision {
    total_queries: usize,
    queries_with_results: usize,
    queries_reranked: usize,
    avg_top_k_overlap: f32,
    median_top_k_overlap: f32,
    pct_better: f32,
    pct_same: f32,
    pct_worse: f32,
    pct_unclear: f32,
    pct_with_movement: f32,
    avg_belief_coverage: f32,
    avg_latency_delta_us: f64,
    max_latency_delta_us: i64,
    pct_contradiction_improved: f32,
    pct_contradiction_worsened: f32,
    avg_max_up_shift: f32,
    avg_max_down_shift: f32,
    max_observed_up_shift: usize,
    max_observed_down_shift: usize,
}

// ─────────────────────────────────────────────────────────
// Collection
// ─────────────────────────────────────────────────────────

fn collect_evidence(
    aura: &Aura,
    queries: &[(&str, usize, &str)], // (query, top_k, category)
) -> Vec<QueryEvidence> {
    queries.iter().map(|(query, top_k, category)| {
        // 1. Baseline: use shadow API (raw baseline, no reranking)
        let baseline_start = std::time::Instant::now();
        let (baseline, shadow_report) = aura
            .recall_structured_with_shadow(
                query, Some(*top_k), Some(0.0), Some(true), None, None,
            )
            .expect("shadow recall failed");
        let baseline_latency = baseline_start.elapsed().as_micros() as u64;

        // 2. Limited: use rerank report API (raw baseline + single rerank pass)
        let limited_start = std::time::Instant::now();
        let (limited, rerank_report) = aura
            .recall_structured_with_rerank_report(
                query, Some(*top_k), Some(0.0), Some(true), None, None,
            )
            .expect("rerank report failed");
        let limited_latency = limited_start.elapsed().as_micros() as u64;

        let baseline_ids: Vec<String> = baseline.iter().map(|(_, r)| r.id.clone()).collect();
        let limited_ids: Vec<String> = limited.iter().map(|(_, r)| r.id.clone()).collect();

        // Top-k overlap (computed manually since both are independent calls)
        let effective_k = baseline_ids.len().min(*top_k);
        let baseline_top: HashSet<&str> = baseline_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
        let limited_top: HashSet<&str> = limited_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
        let overlap = if effective_k > 0 {
            baseline_top.intersection(&limited_top).count() as f32 / effective_k as f32
        } else {
            1.0
        };

        // Contradiction tracking: count unresolved-belief records in each top-k
        // We use shadow_report to identify unresolved records
        let unresolved_ids: HashSet<&str> = shadow_report.scores.iter()
            .filter(|s| s.belief_state.as_deref() == Some("unresolved"))
            .map(|s| s.record_id.as_str())
            .collect();

        let baseline_unresolved = baseline_ids.iter()
            .take(effective_k)
            .filter(|id| unresolved_ids.contains(id.as_str()))
            .count();
        let limited_unresolved = limited_ids.iter()
            .take(effective_k)
            .filter(|id| unresolved_ids.contains(id.as_str()))
            .count();
        let contradiction_delta = limited_unresolved as i32 - baseline_unresolved as i32;

        // Heuristic quality label
        let label = compute_quality_label(
            overlap,
            rerank_report.was_applied,
            rerank_report.records_moved,
            contradiction_delta,
            shadow_report.belief_coverage,
        );

        QueryEvidence {
            query: query.to_string(),
            category: category.to_string(),
            top_k: *top_k,
            baseline_count: baseline.len(),
            limited_count: limited.len(),
            baseline_ids,
            limited_ids,
            top_k_overlap: overlap,
            records_moved: rerank_report.records_moved,
            max_up_shift: rerank_report.max_up_shift,
            max_down_shift: rerank_report.max_down_shift,
            belief_coverage: rerank_report.belief_coverage,
            avg_belief_multiplier: rerank_report.avg_belief_multiplier,
            baseline_latency_us: baseline_latency,
            limited_latency_us: limited_latency,
            latency_delta_us: limited_latency as i64 - baseline_latency as i64,
            rerank_was_applied: rerank_report.was_applied,
            skip_reason: rerank_report.skip_reason.clone(),
            baseline_unresolved_in_top_k: baseline_unresolved,
            limited_unresolved_in_top_k: limited_unresolved,
            contradiction_delta,
            label,
        }
    }).collect()
}

/// Heuristic quality label assignment.
///
/// Not just "did it move" — evaluates whether movement is beneficial:
/// - BETTER: movement present, no contradiction increase, decent coverage
/// - SAME: no movement or rerank not applied
/// - WORSE: contradiction leakage increased OR overlap dropped too low
/// - UNCLEAR: movement but ambiguous quality
fn compute_quality_label(
    overlap: f32,
    was_applied: bool,
    records_moved: usize,
    contradiction_delta: i32,
    belief_coverage: f32,
) -> QualityLabel {
    if !was_applied || records_moved == 0 {
        return QualityLabel::Same;
    }

    // Contradiction increased → worse
    if contradiction_delta > 0 {
        return QualityLabel::Worse;
    }

    // Overlap too low → worse (reranking is too aggressive)
    if overlap < 0.5 {
        return QualityLabel::Worse;
    }

    // Contradiction decreased → better (pushed unresolved records down)
    if contradiction_delta < 0 {
        return QualityLabel::Better;
    }

    // Movement with decent coverage and stable overlap → better
    if belief_coverage >= 0.10 && overlap >= 0.80 {
        return QualityLabel::Better;
    }

    // Movement but low coverage — can't tell if useful
    if belief_coverage < 0.05 {
        return QualityLabel::Unclear;
    }

    // Moderate overlap, some coverage → unclear
    QualityLabel::Unclear
}

// ─────────────────────────────────────────────────────────
// Aggregation
// ─────────────────────────────────────────────────────────

fn compute_decision(evidence: &[QueryEvidence]) -> AggregateDecision {
    let total = evidence.len();
    let with_results: Vec<&QueryEvidence> = evidence.iter()
        .filter(|e| e.baseline_count > 0)
        .collect();
    let n = with_results.len();

    let reranked: Vec<&&QueryEvidence> = with_results.iter()
        .filter(|e| e.rerank_was_applied)
        .collect();
    let nr = reranked.len();

    // Overlap
    let mut overlaps: Vec<f32> = with_results.iter().map(|e| e.top_k_overlap).collect();
    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_overlap = if n > 0 { overlaps.iter().sum::<f32>() / n as f32 } else { 1.0 };
    let median_overlap = if n > 0 {
        if n % 2 == 0 { (overlaps[n/2 - 1] + overlaps[n/2]) / 2.0 } else { overlaps[n/2] }
    } else { 1.0 };

    // Quality labels
    let count_label = |l: QualityLabel| with_results.iter().filter(|e| e.label == l).count();
    let better = count_label(QualityLabel::Better);
    let same = count_label(QualityLabel::Same);
    let worse = count_label(QualityLabel::Worse);
    let unclear = count_label(QualityLabel::Unclear);

    let with_movement = with_results.iter().filter(|e| e.records_moved > 0).count();

    // Coverage
    let avg_coverage = if n > 0 {
        with_results.iter().map(|e| e.belief_coverage).sum::<f32>() / n as f32
    } else { 0.0 };

    // Latency delta
    let avg_latency_delta = if n > 0 {
        with_results.iter().map(|e| e.latency_delta_us).sum::<i64>() as f64 / n as f64
    } else { 0.0 };
    let max_latency_delta = with_results.iter()
        .map(|e| e.latency_delta_us)
        .max()
        .unwrap_or(0);

    // Contradiction improvement
    let contradiction_improved = with_results.iter()
        .filter(|e| e.contradiction_delta < 0)
        .count();
    let contradiction_worsened = with_results.iter()
        .filter(|e| e.contradiction_delta > 0)
        .count();

    // Positional shifts
    let avg_up = if nr > 0 {
        reranked.iter().map(|e| e.max_up_shift as f32).sum::<f32>() / nr as f32
    } else { 0.0 };
    let avg_down = if nr > 0 {
        reranked.iter().map(|e| e.max_down_shift as f32).sum::<f32>() / nr as f32
    } else { 0.0 };
    let max_up = with_results.iter().map(|e| e.max_up_shift).max().unwrap_or(0);
    let max_down = with_results.iter().map(|e| e.max_down_shift).max().unwrap_or(0);

    AggregateDecision {
        total_queries: total,
        queries_with_results: n,
        queries_reranked: nr,
        avg_top_k_overlap: avg_overlap,
        median_top_k_overlap: median_overlap,
        pct_better: if n > 0 { better as f32 / n as f32 } else { 0.0 },
        pct_same: if n > 0 { same as f32 / n as f32 } else { 0.0 },
        pct_worse: if n > 0 { worse as f32 / n as f32 } else { 0.0 },
        pct_unclear: if n > 0 { unclear as f32 / n as f32 } else { 0.0 },
        pct_with_movement: if n > 0 { with_movement as f32 / n as f32 } else { 0.0 },
        avg_belief_coverage: avg_coverage,
        avg_latency_delta_us: avg_latency_delta,
        max_latency_delta_us: max_latency_delta,
        pct_contradiction_improved: if n > 0 { contradiction_improved as f32 / n as f32 } else { 0.0 },
        pct_contradiction_worsened: if n > 0 { contradiction_worsened as f32 / n as f32 } else { 0.0 },
        avg_max_up_shift: avg_up,
        avg_max_down_shift: avg_down,
        max_observed_up_shift: max_up,
        max_observed_down_shift: max_down,
    }
}

// ─────────────────────────────────────────────────────────
// Corpus
// ─────────────────────────────────────────────────────────

fn build_evidence_corpus(aura: &Aura) {
    // Domain 1: Rust programming (strong belief candidates)
    store_batch(aura, &[
        ("Rust is the best language for systems programming", Level::Domain, &["rust", "programming", "systems"], "recorded", "preference"),
        ("Rust memory safety prevents buffer overflows", Level::Domain, &["rust", "programming", "safety"], "recorded", "fact"),
        ("Rust borrow checker catches data races at compile time", Level::Domain, &["rust", "programming", "safety"], "recorded", "fact"),
        ("I always choose Rust for new backend services", Level::Domain, &["rust", "programming", "backend"], "recorded", "preference"),
        ("Rust async ecosystem has matured significantly with tokio", Level::Domain, &["rust", "programming", "async", "tokio"], "recorded", "fact"),
        ("Rust zero-cost abstractions avoid runtime overhead", Level::Domain, &["rust", "programming", "performance"], "recorded", "fact"),
        ("Rust cargo package manager is best in class", Level::Domain, &["rust", "programming", "cargo"], "recorded", "fact"),
        ("Rust error handling with Result type is explicit and safe", Level::Domain, &["rust", "programming", "errors"], "recorded", "fact"),
    ]);

    // Domain 2: Python ecosystem
    store_batch(aura, &[
        ("Python is easier to learn than compiled languages", Level::Domain, &["python", "programming", "learning"], "recorded", "fact"),
        ("Python data science ecosystem with numpy pandas is unmatched", Level::Domain, &["python", "programming", "data", "numpy"], "recorded", "fact"),
        ("I use Python for quick prototyping and scripting tasks", Level::Domain, &["python", "programming", "scripting"], "recorded", "preference"),
        ("Python machine learning libraries like pytorch are excellent", Level::Domain, &["python", "programming", "ml", "pytorch"], "recorded", "fact"),
        ("Python GIL limits true parallelism in CPU-bound tasks", Level::Domain, &["python", "programming", "concurrency", "gil"], "recorded", "fact"),
        ("Python type hints improve code readability", Level::Domain, &["python", "programming", "types"], "recorded", "fact"),
    ]);

    // Domain 3: TypeScript / web
    store_batch(aura, &[
        ("TypeScript adds type safety to JavaScript projects", Level::Domain, &["typescript", "programming", "web"], "recorded", "fact"),
        ("TypeScript catches errors before runtime in web apps", Level::Domain, &["typescript", "programming", "safety", "web"], "recorded", "fact"),
        ("React with TypeScript is the standard for frontend", Level::Domain, &["typescript", "react", "web", "frontend"], "recorded", "fact"),
        ("Next.js simplifies server-side rendering in React", Level::Domain, &["typescript", "nextjs", "web", "ssr"], "recorded", "fact"),
    ]);

    // Domain 4: DevOps and deployment (causal pattern candidates)
    store_batch(aura, &[
        ("Deployed canary releases to production environment successfully", Level::Domain, &["deploy", "canary", "ops"], "recorded", "decision"),
        ("Canary deployment caught critical regression before full rollout", Level::Domain, &["deploy", "canary", "ops", "regression"], "recorded", "fact"),
        ("Blue-green deployment reduces downtime during releases", Level::Domain, &["deploy", "blue-green", "ops"], "recorded", "fact"),
        ("Feature flags decouple deployment from feature release", Level::Domain, &["deploy", "feature-flags", "ops"], "recorded", "decision"),
        ("CI/CD pipeline runs all tests before deployment automatically", Level::Domain, &["deploy", "ci-cd", "testing", "automation"], "recorded", "fact"),
        ("Rollback procedure must be tested before each major release", Level::Domain, &["deploy", "rollback", "ops", "safety"], "recorded", "decision"),
        ("Monitoring alerts should fire within two minutes of anomaly", Level::Domain, &["monitoring", "alerting", "ops"], "recorded", "fact"),
        ("Infrastructure as code prevents configuration drift in production", Level::Domain, &["infra", "iac", "ops", "terraform"], "recorded", "fact"),
        ("Kubernetes self-healing restarts failed pods automatically", Level::Domain, &["infra", "kubernetes", "ops", "resilience"], "recorded", "fact"),
        ("Container images should be immutable and versioned", Level::Domain, &["infra", "docker", "ops", "containers"], "recorded", "fact"),
    ]);

    // Domain 5: Architecture patterns
    store_batch(aura, &[
        ("Microservices enable independent team scaling and deployment", Level::Domain, &["architecture", "microservices"], "recorded", "fact"),
        ("Event sourcing provides complete audit trail of state changes", Level::Domain, &["architecture", "event-sourcing", "audit"], "recorded", "fact"),
        ("CQRS separates read and write models for better scalability", Level::Domain, &["architecture", "cqrs", "scalability"], "recorded", "fact"),
        ("Domain driven design aligns code structure with business domain", Level::Domain, &["architecture", "ddd", "design"], "recorded", "fact"),
        ("API gateway pattern simplifies client integration", Level::Domain, &["architecture", "api-gateway", "integration"], "recorded", "fact"),
        ("Circuit breaker pattern prevents cascade failures in distributed systems", Level::Domain, &["architecture", "resilience", "circuit-breaker"], "recorded", "fact"),
        ("Saga pattern manages distributed transactions without two-phase commit", Level::Domain, &["architecture", "saga", "transactions"], "recorded", "fact"),
        ("Hexagonal architecture isolates domain logic from infrastructure", Level::Domain, &["architecture", "hexagonal", "clean"], "recorded", "fact"),
    ]);

    // Domain 6: Database decisions
    store_batch(aura, &[
        ("PostgreSQL is the default choice for relational data", Level::Domain, &["database", "postgresql", "relational"], "recorded", "preference"),
        ("Redis caching reduced API latency by seventy percent", Level::Domain, &["database", "redis", "caching", "performance"], "recorded", "fact"),
        ("MongoDB works well for document-oriented workloads", Level::Domain, &["database", "mongodb", "document"], "recorded", "fact"),
        ("Database migrations must be backward compatible for zero-downtime", Level::Domain, &["database", "migrations", "safety"], "recorded", "decision"),
        ("Connection pooling is essential for production database performance", Level::Domain, &["database", "performance", "pooling"], "recorded", "fact"),
        ("SQLite is ideal for embedded and local-first applications", Level::Domain, &["database", "sqlite", "embedded"], "recorded", "fact"),
        ("Database indexes should cover all frequent query patterns", Level::Domain, &["database", "indexes", "performance"], "recorded", "fact"),
    ]);

    // Domain 7: Personal workflow / identity
    store_batch(aura, &[
        ("I prefer dark mode in all development tools and editors", Level::Identity, &["preference", "editor", "dark-mode"], "recorded", "preference"),
        ("Neovim is my primary code editor for all projects", Level::Identity, &["preference", "editor", "neovim"], "recorded", "preference"),
        ("I review pull requests every morning before coding sessions", Level::Identity, &["workflow", "code-review", "morning"], "recorded", "preference"),
        ("Deep work blocks of two hours are most productive for me", Level::Identity, &["workflow", "productivity", "focus"], "recorded", "preference"),
        ("I always write tests before implementing features in TDD style", Level::Identity, &["workflow", "tdd", "testing"], "recorded", "preference"),
        ("I use pomodoro technique for focused coding sessions", Level::Identity, &["workflow", "pomodoro", "productivity"], "recorded", "preference"),
    ]);

    // Domain 8: Contested / contradictory claims (should form unresolved beliefs)
    store_batch(aura, &[
        ("Tabs are better than spaces for indentation in code", Level::Working, &["style", "formatting", "indentation"], "recorded", "preference"),
        ("Spaces are better than tabs for consistent code alignment", Level::Working, &["style", "formatting", "indentation"], "recorded", "preference"),
        ("ORM is necessary for database abstraction and portability", Level::Working, &["database", "orm", "abstraction"], "recorded", "preference"),
        ("Raw SQL is better than ORM for complex query performance", Level::Working, &["database", "orm", "sql", "performance"], "recorded", "preference"),
        ("Monorepo is better for code sharing across teams", Level::Working, &["repository", "monorepo", "organization"], "recorded", "preference"),
        ("Polyrepo is better for team autonomy and independence", Level::Working, &["repository", "polyrepo", "organization"], "recorded", "preference"),
        ("GraphQL is superior to REST for frontend flexibility", Level::Working, &["api", "graphql", "frontend"], "recorded", "preference"),
        ("REST is simpler and more cacheable than GraphQL", Level::Working, &["api", "rest", "simplicity"], "recorded", "preference"),
    ]);

    // Domain 9: Repeated facts (strong singleton belief candidates)
    for i in 0..8 {
        aura.store(
            &format!("The team uses trunk-based development as the primary git workflow iteration {}", i),
            Some(Level::Domain),
            Some(vec!["workflow".to_string(), "git".to_string(), "trunk".to_string(), "branching".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }

    // Domain 10: Security practices
    store_batch(aura, &[
        ("All API endpoints must validate input before processing", Level::Domain, &["security", "api", "validation"], "recorded", "decision"),
        ("Secrets must never be committed to version control", Level::Domain, &["security", "secrets", "git"], "recorded", "decision"),
        ("Dependencies should be audited regularly for vulnerabilities", Level::Domain, &["security", "dependencies", "audit"], "recorded", "decision"),
        ("HTTPS is mandatory for all production endpoints", Level::Domain, &["security", "https", "production"], "recorded", "decision"),
        ("Authentication tokens must have short expiry times", Level::Domain, &["security", "auth", "tokens"], "recorded", "decision"),
    ]);
}

// ─────────────────────────────────────────────────────────
// Query pack: 80+ queries across categories
// ─────────────────────────────────────────────────────────

fn query_pack() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        // ── Stable factual recall (should mostly be Same) ──
        ("Rust programming language", 10, "stable-factual"),
        ("Rust memory safety borrow checker", 10, "stable-factual"),
        ("Rust async tokio ecosystem", 10, "stable-factual"),
        ("Rust zero-cost abstractions", 5, "stable-factual"),
        ("Rust cargo package manager", 10, "stable-factual"),
        ("Rust error handling Result", 10, "stable-factual"),
        ("Python data science numpy pandas", 10, "stable-factual"),
        ("Python machine learning pytorch", 10, "stable-factual"),
        ("Python GIL concurrency", 10, "stable-factual"),
        ("TypeScript type safety web", 10, "stable-factual"),
        ("React TypeScript frontend", 10, "stable-factual"),

        // ── Belief-heavy topics (beliefs should influence) ──
        ("best language for backend systems", 10, "belief-heavy"),
        ("programming language choice for new project", 20, "belief-heavy"),
        ("type safety in programming languages", 10, "belief-heavy"),
        ("canary deployment vs blue-green", 10, "belief-heavy"),
        ("deployment strategy for production", 20, "belief-heavy"),
        ("database choice relational vs document", 10, "belief-heavy"),
        ("PostgreSQL vs MongoDB for new service", 10, "belief-heavy"),
        ("architecture pattern for microservices", 20, "belief-heavy"),
        ("testing strategy test-driven development", 10, "belief-heavy"),
        ("code editor preference neovim", 10, "belief-heavy"),

        // ── Conflicting recall (unresolved beliefs — rerank should penalize) ──
        ("tabs vs spaces indentation", 10, "conflicting"),
        ("ORM vs raw SQL performance", 10, "conflicting"),
        ("monorepo vs polyrepo organization", 10, "conflicting"),
        ("GraphQL vs REST API design", 10, "conflicting"),
        ("indentation style for code formatting", 10, "conflicting"),
        ("database abstraction layer ORM", 10, "conflicting"),
        ("repository structure monorepo polyrepo", 10, "conflicting"),
        ("API design GraphQL REST", 10, "conflicting"),

        // ── Low-belief-coverage topics (rerank should skip) ──
        ("SQLite embedded local-first", 5, "low-coverage"),
        ("database indexes query performance", 10, "low-coverage"),
        ("pomodoro technique coding", 5, "low-coverage"),
        ("saga distributed transactions", 10, "low-coverage"),
        ("hexagonal architecture clean", 10, "low-coverage"),
        ("Next.js server-side rendering", 5, "low-coverage"),

        // ── DevOps and deployment (mixed coverage) ──
        ("canary deployment production", 10, "devops"),
        ("feature flags deployment release", 10, "devops"),
        ("CI CD pipeline testing automation", 10, "devops"),
        ("rollback procedure safety", 5, "devops"),
        ("monitoring alerting anomaly detection", 10, "devops"),
        ("infrastructure as code terraform", 10, "devops"),
        ("kubernetes pod self-healing", 10, "devops"),
        ("container images docker immutable", 10, "devops"),

        // ── Architecture ──
        ("microservices independent scaling", 10, "architecture"),
        ("event sourcing audit trail", 10, "architecture"),
        ("CQRS read write model", 10, "architecture"),
        ("domain driven design business", 10, "architecture"),
        ("API gateway pattern integration", 5, "architecture"),
        ("circuit breaker cascade failure", 10, "architecture"),

        // ── Database ──
        ("PostgreSQL relational data", 10, "database"),
        ("Redis caching API latency", 10, "database"),
        ("MongoDB document workload", 10, "database"),
        ("database migrations backward compatible", 10, "database"),
        ("connection pooling production", 5, "database"),

        // ── Personal workflow ──
        ("dark mode development tools", 10, "workflow"),
        ("neovim primary editor", 5, "workflow"),
        ("pull request code review morning", 10, "workflow"),
        ("deep work productivity focus", 10, "workflow"),
        ("test driven development TDD", 10, "workflow"),

        // ── Security ──
        ("API input validation security", 10, "security"),
        ("secrets version control git", 10, "security"),
        ("dependency audit vulnerabilities", 10, "security"),
        ("HTTPS production endpoints mandatory", 10, "security"),
        ("authentication tokens expiry", 10, "security"),

        // ── Cross-domain queries (mixed signals) ──
        ("testing deployment pipeline safety", 10, "cross-domain"),
        ("production safety monitoring strategy", 10, "cross-domain"),
        ("backend service architecture performance", 10, "cross-domain"),
        ("development tools workflow preference", 10, "cross-domain"),
        ("performance optimization caching database", 10, "cross-domain"),
        ("security audit deployment pipeline", 10, "cross-domain"),
        ("code quality testing review", 10, "cross-domain"),

        // ── Repeated facts (strong singleton signals) ──
        ("trunk-based development git workflow", 10, "repeated"),
        ("team git branching strategy", 10, "repeated"),
        ("trunk development branching model", 5, "repeated"),

        // ── Broad queries (large result set) ──
        ("best practices software engineering", 20, "broad"),
        ("how to deploy safely to production", 20, "broad"),
        ("what programming language to use", 20, "broad"),
        ("software architecture design patterns", 20, "broad"),

        // ── Edge cases: no matches expected (rerank should skip) ──
        ("quantum computing blockchain integration", 10, "no-match"),
        ("gardening tips for spring season", 5, "no-match"),
        ("recipe for chocolate cake baking", 10, "no-match"),
        ("weather forecast tomorrow", 5, "no-match"),
    ]
}

// ═════════════════════════════════════════════════════════
// EVIDENCE COLLECTION TEST
// ═════════════════════════════════════════════════════════

#[test]
fn phase4_evidence_collection() {
    let (aura, _dir) = open_temp_aura();

    // Build corpus
    build_evidence_corpus(&aura);

    // Run enough maintenance cycles for beliefs to form and stabilize
    run_cycles(&aura, 12);

    // Collect evidence
    let queries = query_pack();
    assert!(queries.len() >= 80, "need >= 80 queries, got {}", queries.len());

    let evidence = collect_evidence(&aura, &queries);
    let agg = compute_decision(&evidence);

    // ── Print detailed report ──
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("PHASE 4 EVIDENCE REPORT — BASELINE vs LIMITED RERANK");
    eprintln!("{}", "=".repeat(80));
    eprintln!("Total queries:          {}", agg.total_queries);
    eprintln!("Queries with results:   {}", agg.queries_with_results);
    eprintln!("Queries reranked:       {}", agg.queries_reranked);
    eprintln!();
    eprintln!("── Overlap ──");
    eprintln!("  Avg top-k overlap:    {:.3}", agg.avg_top_k_overlap);
    eprintln!("  Median top-k overlap: {:.3}", agg.median_top_k_overlap);
    eprintln!();
    eprintln!("── Quality Labels ──");
    eprintln!("  BETTER:   {:.1}%", agg.pct_better * 100.0);
    eprintln!("  SAME:     {:.1}%", agg.pct_same * 100.0);
    eprintln!("  WORSE:    {:.1}%", agg.pct_worse * 100.0);
    eprintln!("  UNCLEAR:  {:.1}%", agg.pct_unclear * 100.0);
    eprintln!();
    eprintln!("── Movement ──");
    eprintln!("  % with movement:      {:.1}%", agg.pct_with_movement * 100.0);
    eprintln!("  Avg belief coverage:  {:.3}", agg.avg_belief_coverage);
    eprintln!("  Avg max up shift:     {:.1}", agg.avg_max_up_shift);
    eprintln!("  Avg max down shift:   {:.1}", agg.avg_max_down_shift);
    eprintln!("  Max observed up:      {}", agg.max_observed_up_shift);
    eprintln!("  Max observed down:    {}", agg.max_observed_down_shift);
    eprintln!();
    eprintln!("── Contradiction Leakage ──");
    eprintln!("  % improved:           {:.1}%", agg.pct_contradiction_improved * 100.0);
    eprintln!("  % worsened:           {:.1}%", agg.pct_contradiction_worsened * 100.0);
    eprintln!();
    eprintln!("── Latency ──");
    eprintln!("  Avg delta:            {:.0}μs", agg.avg_latency_delta_us);
    eprintln!("  Max delta:            {}μs", agg.max_latency_delta_us);
    eprintln!();

    // Per-category breakdown
    let categories = [
        "stable-factual", "belief-heavy", "conflicting", "low-coverage",
        "devops", "architecture", "database", "workflow", "security",
        "cross-domain", "repeated", "broad", "no-match",
    ];
    eprintln!("── Per-Category Summary ──");
    for cat in &categories {
        let cat_evidence: Vec<&QueryEvidence> = evidence.iter()
            .filter(|e| e.category == *cat)
            .collect();
        if cat_evidence.is_empty() { continue; }

        let n = cat_evidence.len();
        let with_results = cat_evidence.iter().filter(|e| e.baseline_count > 0).count();
        let reranked = cat_evidence.iter().filter(|e| e.rerank_was_applied).count();
        let moved = cat_evidence.iter().filter(|e| e.records_moved > 0).count();
        let better = cat_evidence.iter().filter(|e| e.label == QualityLabel::Better).count();
        let worse = cat_evidence.iter().filter(|e| e.label == QualityLabel::Worse).count();

        eprintln!(
            "  {:<16} n={:>2}  results={:>2}  reranked={:>2}  moved={:>2}  better={:>2}  worse={:>2}",
            cat, n, with_results, reranked, moved, better, worse,
        );
    }
    eprintln!();

    // Per-query detail for queries with movement
    let mut moved: Vec<&QueryEvidence> = evidence.iter()
        .filter(|e| e.records_moved > 0 && e.baseline_count > 0)
        .collect();
    moved.sort_by(|a, b| a.top_k_overlap.partial_cmp(&b.top_k_overlap).unwrap());

    if !moved.is_empty() {
        eprintln!("── Queries with Movement ({}) ──", moved.len());
        for e in &moved {
            eprintln!(
                "  [{:.2} ovlp] [{:>7}] q=\"{}\" cat={} k={} results={} moved={} +{}/-{} cov={:.2} cdelta={}",
                e.top_k_overlap, format!("{}", e.label), e.query, e.category, e.top_k,
                e.baseline_count, e.records_moved, e.max_up_shift, e.max_down_shift,
                e.belief_coverage, e.contradiction_delta,
            );
        }
        eprintln!();
    }

    // ── Decision thresholds ──
    eprintln!("{}", "=".repeat(80));
    eprintln!("DECISION ASSESSMENT");
    eprintln!("{}", "=".repeat(80));

    // Safety gates (hard assertions)
    let safety_ok = true;

    // Gate 1: No crashes — if we got here, we passed
    eprintln!("  [PASS] No crashes — all {} queries completed", agg.total_queries);

    // Gate 2: Overlap stability
    let overlap_ok = agg.avg_top_k_overlap >= 0.70;
    eprintln!("  [{}] Avg overlap >= 0.70: {:.3}",
        if overlap_ok { "PASS" } else { "FAIL" }, agg.avg_top_k_overlap);

    // Gate 3: Regressions rare
    let regressions_ok = agg.pct_worse < 0.10;
    eprintln!("  [{}] Regressions < 10%: {:.1}%",
        if regressions_ok { "PASS" } else { "FAIL" }, agg.pct_worse * 100.0);

    // Gate 4: Contradiction leakage not worsened
    let contradiction_ok = agg.pct_contradiction_worsened <= agg.pct_contradiction_improved + 0.05;
    eprintln!("  [{}] Contradiction leakage stable: improved={:.1}% worsened={:.1}%",
        if contradiction_ok { "PASS" } else { "FAIL" },
        agg.pct_contradiction_improved * 100.0, agg.pct_contradiction_worsened * 100.0);

    // Gate 5: Positional shift bounded
    let shift_ok = agg.max_observed_up_shift <= 2 && agg.max_observed_down_shift <= 2;
    eprintln!("  [{}] Positional shift <= 2: max_up={} max_down={}",
        if shift_ok { "PASS" } else { "FAIL" }, agg.max_observed_up_shift, agg.max_observed_down_shift);

    // Gate 6: Latency budget
    let latency_ok = agg.max_latency_delta_us < 2_000_000; // 2 seconds total delta
    eprintln!("  [{}] Latency budget: max_delta={}μs",
        if latency_ok { "PASS" } else { "FAIL" }, agg.max_latency_delta_us);

    eprintln!();

    // ── Final verdict ──
    let all_gates = safety_ok && overlap_ok && regressions_ok && contradiction_ok && shift_ok && latency_ok;

    let verdict = if !all_gates {
        "ROLLBACK TO SHADOW — safety gates failed"
    } else if agg.pct_better >= 0.10 && agg.pct_worse < 0.05 {
        "PREPARE FOR WIDER ROLLOUT — clear benefit, minimal risk"
    } else if agg.pct_better > 0.0 || agg.pct_with_movement > 0.10 {
        "KEEP LIMITED EXPERIMENTAL — some signal, needs more observation"
    } else {
        "KEEP LIMITED EXPERIMENTAL — safe but no clear benefit yet"
    };

    eprintln!("VERDICT: {}", verdict);
    eprintln!();

    // ── Hard assertions ──
    assert!(agg.avg_top_k_overlap >= 0.70,
        "avg overlap {:.3} < 0.70", agg.avg_top_k_overlap);
    assert!(agg.pct_worse < 0.15,
        "regressions {:.1}% >= 15%", agg.pct_worse * 100.0);
    assert!(agg.max_observed_up_shift <= 2,
        "max up shift {} > 2", agg.max_observed_up_shift);
    assert!(agg.max_observed_down_shift <= 2,
        "max down shift {} > 2", agg.max_observed_down_shift);
    assert!(agg.queries_with_results >= 40,
        "only {} queries returned results", agg.queries_with_results);
}
