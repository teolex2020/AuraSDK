//! Candidate B Steady-State Monitoring Job
//!
//! Run periodically (weekly recommended) to collect operational metrics
//! for the belief reranking feature in wider rollout mode.
//!
//! Collects:
//!   - belief coverage trend
//!   - movement rate
//!   - % better / same / worse
//!   - contradiction leakage
//!   - latency drift
//!   - scope-guard skip rate
//!   - positional shift distribution
//!
//! Asserts alert thresholds defined in BELIEF_RERANK_MONITORING.md.
//!
//! Usage:
//!   cargo test --no-default-features --features "encryption,server,audit" \
//!     --test belief_rerank_monitor -- --nocapture

use aura::{Aura, Level};
use std::collections::HashSet;
use std::mem::ManuallyDrop;

// ─────────────────────────────────────────────────────────
// Alert thresholds (must match BELIEF_RERANK_MONITORING.md)
// ─────────────────────────────────────────────────────────

/// Maximum allowed % of WORSE queries. Alert if exceeded.
const ALERT_MAX_WORSE_PCT: f32 = 0.05;
/// Minimum avg top-k overlap. Alert if below.
const ALERT_MIN_AVG_OVERLAP: f32 = 0.70;
/// Maximum positional shift (hard cap). Alert if exceeded.
const ALERT_MAX_POS_SHIFT: usize = 2;
/// Maximum avg latency delta in microseconds. Alert if exceeded.
const ALERT_MAX_AVG_LATENCY_US: f64 = 2000.0;
/// Maximum contradiction worsened %. Alert if exceeded.
const ALERT_MAX_CONTRADICTION_WORSENED_PCT: f32 = 0.05;
/// Minimum belief coverage (trend alert, not hard block).
const ALERT_MIN_BELIEF_COVERAGE: f32 = 0.01;
/// Maximum scope-guard skip rate (too many skips = coverage problem).
const ALERT_MAX_SKIP_RATE: f32 = 0.95;

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

struct TempAura {
    aura: ManuallyDrop<Aura>,
    dir: ManuallyDrop<tempfile::TempDir>,
}

impl std::ops::Deref for TempAura {
    type Target = Aura;

    fn deref(&self) -> &Self::Target {
        &self.aura
    }
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
    TempAura {
        aura: ManuallyDrop::new(aura),
        dir: ManuallyDrop::new(dir),
    }
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

// ─────────────────────────────────────────────────────────
// Per-query metrics
// ─────────────────────────────────────────────────────────

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

#[allow(dead_code)]
struct QueryMetrics {
    query: String,
    category: String,
    baseline_count: usize,
    limited_count: usize,
    top_k_overlap: f32,
    records_moved: usize,
    max_up_shift: usize,
    max_down_shift: usize,
    belief_coverage: f32,
    baseline_latency_us: u64,
    limited_latency_us: u64,
    latency_delta_us: i64,
    rerank_was_applied: bool,
    skip_reason: String,
    contradiction_delta: i32,
    label: QualityLabel,
}

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
    if contradiction_delta > 0 { return QualityLabel::Worse; }
    if overlap < 0.5 { return QualityLabel::Worse; }
    if contradiction_delta < 0 { return QualityLabel::Better; }
    if belief_coverage >= 0.10 && overlap >= 0.80 { return QualityLabel::Better; }
    if belief_coverage < 0.05 { return QualityLabel::Unclear; }
    QualityLabel::Unclear
}

fn collect_metrics(
    aura: &Aura,
    queries: &[(&str, usize, &str)],
) -> Vec<QueryMetrics> {
    queries.iter().map(|(query, top_k, category)| {
        // Baseline: shadow API (raw, no rerank)
        let baseline_start = std::time::Instant::now();
        let (baseline, shadow_report) = aura
            .recall_structured_with_shadow(
                query, Some(*top_k), Some(0.0), Some(true), None, None,
            )
            .expect("shadow recall failed");
        let baseline_latency = baseline_start.elapsed().as_micros() as u64;

        // Limited: rerank report API (raw + single rerank pass)
        let limited_start = std::time::Instant::now();
        let (limited, rerank_report) = aura
            .recall_structured_with_rerank_report(
                query, Some(*top_k), Some(0.0), Some(true), None, None,
            )
            .expect("rerank report failed");
        let limited_latency = limited_start.elapsed().as_micros() as u64;

        let baseline_ids: Vec<String> = baseline.iter().map(|(_, r)| r.id.clone()).collect();
        let limited_ids: Vec<String> = limited.iter().map(|(_, r)| r.id.clone()).collect();

        let effective_k = baseline_ids.len().min(*top_k);
        let baseline_top: HashSet<&str> = baseline_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
        let limited_top: HashSet<&str> = limited_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
        let overlap = if effective_k > 0 {
            baseline_top.intersection(&limited_top).count() as f32 / effective_k as f32
        } else { 1.0 };

        // Contradiction tracking
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

        let label = compute_quality_label(
            overlap,
            rerank_report.was_applied,
            rerank_report.records_moved,
            contradiction_delta,
            shadow_report.belief_coverage,
        );

        QueryMetrics {
            query: query.to_string(),
            category: category.to_string(),
            baseline_count: baseline.len(),
            limited_count: limited.len(),
            top_k_overlap: overlap,
            records_moved: rerank_report.records_moved,
            max_up_shift: rerank_report.max_up_shift,
            max_down_shift: rerank_report.max_down_shift,
            belief_coverage: rerank_report.belief_coverage,
            baseline_latency_us: baseline_latency,
            limited_latency_us: limited_latency,
            latency_delta_us: limited_latency as i64 - baseline_latency as i64,
            rerank_was_applied: rerank_report.was_applied,
            skip_reason: rerank_report.skip_reason.clone(),
            contradiction_delta,
            label,
        }
    }).collect()
}

// ─────────────────────────────────────────────────────────
// Aggregate report
// ─────────────────────────────────────────────────────────

struct MonitorReport {
    total_queries: usize,
    queries_with_results: usize,
    queries_reranked: usize,
    queries_skipped: usize,
    skip_rate: f32,
    avg_top_k_overlap: f32,
    pct_better: f32,
    pct_same: f32,
    pct_worse: f32,
    pct_unclear: f32,
    pct_with_movement: f32,
    avg_belief_coverage: f32,
    avg_latency_delta_us: f64,
    max_latency_delta_us: i64,
    pct_contradiction_worsened: f32,
    max_up_shift: usize,
    max_down_shift: usize,
}

fn compute_report(metrics: &[QueryMetrics]) -> MonitorReport {
    let total = metrics.len();
    let with_results: Vec<&QueryMetrics> = metrics.iter()
        .filter(|m| m.baseline_count > 0)
        .collect();
    let n = with_results.len();

    let reranked = with_results.iter().filter(|m| m.rerank_was_applied).count();
    let skipped = n - reranked;
    let skip_rate = if n > 0 { skipped as f32 / n as f32 } else { 0.0 };

    let avg_overlap = if n > 0 {
        with_results.iter().map(|m| m.top_k_overlap).sum::<f32>() / n as f32
    } else { 1.0 };

    let count_label = |l: QualityLabel| with_results.iter().filter(|m| m.label == l).count();
    let better = count_label(QualityLabel::Better);
    let same = count_label(QualityLabel::Same);
    let worse = count_label(QualityLabel::Worse);
    let unclear = count_label(QualityLabel::Unclear);
    let with_movement = with_results.iter().filter(|m| m.records_moved > 0).count();

    let avg_coverage = if n > 0 {
        with_results.iter().map(|m| m.belief_coverage).sum::<f32>() / n as f32
    } else { 0.0 };

    let avg_latency = if n > 0 {
        with_results.iter().map(|m| m.latency_delta_us).sum::<i64>() as f64 / n as f64
    } else { 0.0 };
    let max_latency = with_results.iter()
        .map(|m| m.latency_delta_us)
        .max()
        .unwrap_or(0);

    let contradiction_worsened = with_results.iter()
        .filter(|m| m.contradiction_delta > 0)
        .count();

    let max_up = with_results.iter().map(|m| m.max_up_shift).max().unwrap_or(0);
    let max_down = with_results.iter().map(|m| m.max_down_shift).max().unwrap_or(0);

    MonitorReport {
        total_queries: total,
        queries_with_results: n,
        queries_reranked: reranked,
        queries_skipped: skipped,
        skip_rate,
        avg_top_k_overlap: avg_overlap,
        pct_better: if n > 0 { better as f32 / n as f32 } else { 0.0 },
        pct_same: if n > 0 { same as f32 / n as f32 } else { 0.0 },
        pct_worse: if n > 0 { worse as f32 / n as f32 } else { 0.0 },
        pct_unclear: if n > 0 { unclear as f32 / n as f32 } else { 0.0 },
        pct_with_movement: if n > 0 { with_movement as f32 / n as f32 } else { 0.0 },
        avg_belief_coverage: avg_coverage,
        avg_latency_delta_us: avg_latency,
        max_latency_delta_us: max_latency,
        pct_contradiction_worsened: if n > 0 { contradiction_worsened as f32 / n as f32 } else { 0.0 },
        max_up_shift: max_up,
        max_down_shift: max_down,
    }
}

// ─────────────────────────────────────────────────────────
// Alert evaluation
// ─────────────────────────────────────────────────────────

struct AlertResult {
    name: &'static str,
    threshold: String,
    actual: String,
    passed: bool,
}

fn evaluate_alerts(report: &MonitorReport) -> Vec<AlertResult> {
    vec![
        AlertResult {
            name: "WORSE rate",
            threshold: format!("≤ {:.0}%", ALERT_MAX_WORSE_PCT * 100.0),
            actual: format!("{:.1}%", report.pct_worse * 100.0),
            passed: report.pct_worse <= ALERT_MAX_WORSE_PCT,
        },
        AlertResult {
            name: "Avg top-k overlap",
            threshold: format!("≥ {:.2}", ALERT_MIN_AVG_OVERLAP),
            actual: format!("{:.3}", report.avg_top_k_overlap),
            passed: report.avg_top_k_overlap >= ALERT_MIN_AVG_OVERLAP,
        },
        AlertResult {
            name: "Max positional shift (up)",
            threshold: format!("≤ {}", ALERT_MAX_POS_SHIFT),
            actual: format!("{}", report.max_up_shift),
            passed: report.max_up_shift <= ALERT_MAX_POS_SHIFT,
        },
        AlertResult {
            name: "Max positional shift (down)",
            threshold: format!("≤ {}", ALERT_MAX_POS_SHIFT),
            actual: format!("{}", report.max_down_shift),
            passed: report.max_down_shift <= ALERT_MAX_POS_SHIFT,
        },
        AlertResult {
            name: "Avg latency delta",
            threshold: format!("≤ {:.0}μs", ALERT_MAX_AVG_LATENCY_US),
            actual: format!("{:.0}μs", report.avg_latency_delta_us),
            passed: report.avg_latency_delta_us <= ALERT_MAX_AVG_LATENCY_US,
        },
        AlertResult {
            name: "Contradiction worsened",
            threshold: format!("≤ {:.0}%", ALERT_MAX_CONTRADICTION_WORSENED_PCT * 100.0),
            actual: format!("{:.1}%", report.pct_contradiction_worsened * 100.0),
            passed: report.pct_contradiction_worsened <= ALERT_MAX_CONTRADICTION_WORSENED_PCT,
        },
        AlertResult {
            name: "Belief coverage",
            threshold: format!("≥ {:.0}%", ALERT_MIN_BELIEF_COVERAGE * 100.0),
            actual: format!("{:.1}%", report.avg_belief_coverage * 100.0),
            passed: report.avg_belief_coverage >= ALERT_MIN_BELIEF_COVERAGE,
        },
        AlertResult {
            name: "Scope-guard skip rate",
            threshold: format!("≤ {:.0}%", ALERT_MAX_SKIP_RATE * 100.0),
            actual: format!("{:.1}%", report.skip_rate * 100.0),
            passed: report.skip_rate <= ALERT_MAX_SKIP_RATE,
        },
    ]
}

// ─────────────────────────────────────────────────────────
// Corpus (same as phase4_evidence.rs)
// ─────────────────────────────────────────────────────────

fn build_monitor_corpus(aura: &Aura) {
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
    store_batch(aura, &[
        ("Python is easier to learn than compiled languages", Level::Domain, &["python", "programming", "learning"], "recorded", "fact"),
        ("Python data science ecosystem with numpy pandas is unmatched", Level::Domain, &["python", "programming", "data", "numpy"], "recorded", "fact"),
        ("I use Python for quick prototyping and scripting tasks", Level::Domain, &["python", "programming", "scripting"], "recorded", "preference"),
        ("Python machine learning libraries like pytorch are excellent", Level::Domain, &["python", "programming", "ml", "pytorch"], "recorded", "fact"),
        ("Python GIL limits true parallelism in CPU-bound tasks", Level::Domain, &["python", "programming", "concurrency", "gil"], "recorded", "fact"),
        ("Python type hints improve code readability", Level::Domain, &["python", "programming", "types"], "recorded", "fact"),
    ]);
    store_batch(aura, &[
        ("TypeScript adds type safety to JavaScript projects", Level::Domain, &["typescript", "programming", "web"], "recorded", "fact"),
        ("TypeScript catches errors before runtime in web apps", Level::Domain, &["typescript", "programming", "safety", "web"], "recorded", "fact"),
        ("React with TypeScript is the standard for frontend", Level::Domain, &["typescript", "react", "web", "frontend"], "recorded", "fact"),
        ("Next.js simplifies server-side rendering in React", Level::Domain, &["typescript", "nextjs", "web", "ssr"], "recorded", "fact"),
    ]);
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
    store_batch(aura, &[
        ("Microservices enable independent team scaling and deployment", Level::Domain, &["architecture", "microservices"], "recorded", "fact"),
        ("Event sourcing provides complete audit trail of state changes", Level::Domain, &["architecture", "event-sourcing", "audit"], "recorded", "fact"),
        ("CQRS separates read and write models for better scaling", Level::Domain, &["architecture", "cqrs", "scaling"], "recorded", "fact"),
        ("Domain-driven design aligns code with business boundaries", Level::Domain, &["architecture", "ddd", "business"], "recorded", "fact"),
        ("API gateway pattern simplifies client-server integration", Level::Domain, &["architecture", "api-gateway", "integration"], "recorded", "fact"),
        ("Circuit breaker prevents cascade failures in distributed systems", Level::Domain, &["architecture", "circuit-breaker", "resilience"], "recorded", "fact"),
    ]);
    store_batch(aura, &[
        ("PostgreSQL is the best relational database for production", Level::Domain, &["database", "postgresql", "relational"], "recorded", "preference"),
        ("Redis caching reduces API response latency significantly", Level::Domain, &["database", "redis", "caching", "performance"], "recorded", "fact"),
        ("MongoDB flexible schema works for document-heavy workloads", Level::Domain, &["database", "mongodb", "document"], "recorded", "fact"),
        ("Database migrations must be backward compatible in production", Level::Domain, &["database", "migrations", "safety"], "recorded", "decision"),
        ("Connection pooling is essential for production database", Level::Domain, &["database", "pooling", "production"], "recorded", "fact"),
        ("SQLite works well for embedded local-first applications", Level::Domain, &["database", "sqlite", "embedded"], "recorded", "fact"),
        ("Database indexes should match actual query patterns", Level::Domain, &["database", "indexes", "performance"], "recorded", "fact"),
    ]);
    store_batch(aura, &[
        ("I prefer dark mode for all development tools", Level::Domain, &["workflow", "tools", "preference"], "recorded", "preference"),
        ("Neovim is my primary code editor for all projects", Level::Domain, &["workflow", "editor", "neovim"], "recorded", "preference"),
        ("Code reviews should happen every morning before deep work", Level::Domain, &["workflow", "review", "process"], "recorded", "preference"),
        ("Deep work blocks of four hours improve productivity", Level::Domain, &["workflow", "productivity", "focus"], "recorded", "fact"),
        ("Test-driven development catches bugs before they propagate", Level::Domain, &["workflow", "testing", "tdd"], "recorded", "fact"),
    ]);
    store_batch(aura, &[
        ("All API endpoints must validate input before processing", Level::Domain, &["security", "api", "validation"], "recorded", "decision"),
        ("Never store secrets in version control repositories", Level::Domain, &["security", "secrets", "git"], "recorded", "decision"),
        ("Dependency audit catches known vulnerabilities early", Level::Domain, &["security", "dependencies", "audit"], "recorded", "fact"),
        ("Production endpoints require HTTPS without exceptions", Level::Domain, &["security", "https", "production"], "recorded", "decision"),
        ("Authentication tokens should have short expiry periods", Level::Domain, &["security", "auth", "tokens"], "recorded", "decision"),
    ]);
    // Conflicting data
    store_batch(aura, &[
        ("Tabs are better than spaces for indentation", Level::Domain, &["formatting", "indentation", "preference"], "recorded", "preference"),
        ("Spaces are the standard for code indentation", Level::Domain, &["formatting", "indentation", "preference"], "recorded", "preference"),
        ("ORMs abstract away SQL complexity for faster development", Level::Domain, &["database", "orm", "abstraction"], "recorded", "fact"),
        ("Raw SQL outperforms ORMs for complex database queries", Level::Domain, &["database", "sql", "performance"], "recorded", "fact"),
        ("Monorepos simplify dependency management across teams", Level::Domain, &["vcs", "monorepo", "organization"], "recorded", "fact"),
        ("Polyrepos give teams independent release cycles", Level::Domain, &["vcs", "polyrepo", "organization"], "recorded", "fact"),
        ("GraphQL reduces over-fetching in frontend applications", Level::Domain, &["api", "graphql", "frontend"], "recorded", "fact"),
        ("REST is simpler and better understood than GraphQL", Level::Domain, &["api", "rest", "simplicity"], "recorded", "fact"),
    ]);
    // Repeated/strong signals
    store_batch(aura, &[
        ("Trunk-based development with short-lived feature branches", Level::Domain, &["git", "branching", "trunk-based"], "recorded", "decision"),
        ("Feature branches should merge to trunk within one day", Level::Domain, &["git", "branching", "trunk-based"], "recorded", "decision"),
        ("Trunk-based development reduces integration conflicts", Level::Domain, &["git", "branching", "trunk-based"], "recorded", "fact"),
    ]);
}

// ─────────────────────────────────────────────────────────
// Monitoring query pack (stable — same across runs)
// ─────────────────────────────────────────────────────────

fn monitor_query_pack() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        // Stable factual (expect: Same)
        ("Rust programming language", 10, "stable-factual"),
        ("Rust memory safety borrow checker", 10, "stable-factual"),
        ("Python data science numpy pandas", 10, "stable-factual"),
        ("TypeScript type safety web", 10, "stable-factual"),

        // Belief-heavy (expect: potential movement)
        ("best language for backend systems", 10, "belief-heavy"),
        ("deployment strategy for production", 10, "belief-heavy"),
        ("database choice relational vs document", 10, "belief-heavy"),
        ("code editor preference neovim", 10, "belief-heavy"),

        // Conflicting (expect: unresolved beliefs)
        ("tabs vs spaces indentation", 10, "conflicting"),
        ("ORM vs raw SQL performance", 10, "conflicting"),
        ("GraphQL vs REST API design", 10, "conflicting"),
        ("monorepo vs polyrepo organization", 10, "conflicting"),

        // DevOps (mixed coverage)
        ("canary deployment production", 10, "devops"),
        ("CI CD pipeline testing automation", 10, "devops"),
        ("kubernetes pod self-healing", 10, "devops"),

        // Architecture
        ("microservices independent scaling", 10, "architecture"),
        ("event sourcing audit trail", 10, "architecture"),
        ("circuit breaker cascade failure", 10, "architecture"),

        // Database
        ("PostgreSQL relational data", 10, "database"),
        ("Redis caching API latency", 10, "database"),

        // Workflow
        ("dark mode development tools", 10, "workflow"),
        ("test driven development TDD", 10, "workflow"),

        // Security
        ("API input validation security", 10, "security"),
        ("secrets version control git", 10, "security"),

        // Cross-domain
        ("testing deployment pipeline safety", 10, "cross-domain"),
        ("backend service architecture performance", 10, "cross-domain"),

        // Edge cases
        ("quantum computing blockchain integration", 10, "no-match"),
        ("gardening tips for spring season", 5, "no-match"),
    ]
}

// ═════════════════════════════════════════════════════════
// MONITORING TEST
// ═════════════════════════════════════════════════════════

#[test]
fn belief_rerank_steady_state_monitor() {
    let aura = open_temp_aura();

    // Build corpus
    build_monitor_corpus(&aura);

    // Run maintenance cycles for beliefs to stabilize
    for _ in 0..12 {
        aura.run_maintenance();
    }

    // Collect metrics
    let queries = monitor_query_pack();
    let metrics = collect_metrics(&aura, &queries);
    let report = compute_report(&metrics);
    let alerts = evaluate_alerts(&report);

    // ── Print structured report ──
    eprintln!("\n  ╔═══════════════════════════════════════════════════╗");
    eprintln!("  ║  Candidate B — Steady-State Monitoring Report     ║");
    eprintln!("  ╚═══════════════════════════════════════════════════╝\n");

    eprintln!("  Queries:            {}", report.total_queries);
    eprintln!("  With results:       {}", report.queries_with_results);
    eprintln!("  Reranked:           {}/{}", report.queries_reranked, report.queries_with_results);
    eprintln!("  Skipped:            {} ({:.1}%)", report.queries_skipped, report.skip_rate * 100.0);
    eprintln!("  With movement:      {:.1}%", report.pct_with_movement * 100.0);
    eprintln!();
    eprintln!("  Avg top-k overlap:  {:.3}", report.avg_top_k_overlap);
    eprintln!("  Avg belief coverage:{:.3}", report.avg_belief_coverage);
    eprintln!("  Avg latency delta:  {:.0}μs", report.avg_latency_delta_us);
    eprintln!("  Max latency delta:  {}μs", report.max_latency_delta_us);
    eprintln!("  Max pos shift:      ↑{} ↓{}", report.max_up_shift, report.max_down_shift);
    eprintln!();
    eprintln!("  Quality:  BETTER={:.1}%  SAME={:.1}%  WORSE={:.1}%  UNCLEAR={:.1}%",
        report.pct_better * 100.0, report.pct_same * 100.0,
        report.pct_worse * 100.0, report.pct_unclear * 100.0);
    eprintln!("  Contradiction worsened: {:.1}%", report.pct_contradiction_worsened * 100.0);

    // ── Per-category breakdown ──
    eprintln!("\n  ── Per-Category ──");
    let categories: Vec<&str> = vec![
        "stable-factual", "belief-heavy", "conflicting", "devops",
        "architecture", "database", "workflow", "security", "cross-domain", "no-match",
    ];
    for cat in &categories {
        let cat_metrics: Vec<&QueryMetrics> = metrics.iter()
            .filter(|m| m.category == *cat)
            .collect();
        if cat_metrics.is_empty() { continue; }
        let n = cat_metrics.len();
        let reranked = cat_metrics.iter().filter(|m| m.rerank_was_applied).count();
        let moved = cat_metrics.iter().filter(|m| m.records_moved > 0).count();
        let better = cat_metrics.iter().filter(|m| m.label == QualityLabel::Better).count();
        let worse = cat_metrics.iter().filter(|m| m.label == QualityLabel::Worse).count();
        eprintln!("  {:20} n={:2}  reranked={:2}  moved={:2}  better={:2}  worse={:2}",
            cat, n, reranked, moved, better, worse);
    }

    // ── Alert evaluation ──
    eprintln!("\n  ── Alert Thresholds ──");
    let mut all_passed = true;
    for alert in &alerts {
        let status = if alert.passed { "PASS" } else { "FAIL" };
        let marker = if alert.passed { " " } else { "!" };
        eprintln!("  {} [{}] {:30} threshold: {:>10}  actual: {:>10}",
            marker, status, alert.name, alert.threshold, alert.actual);
        if !alert.passed { all_passed = false; }
    }

    eprintln!();
    if all_passed {
        eprintln!("  ✓ ALL ALERTS PASS — system healthy");
    } else {
        eprintln!("  ✗ ALERT(S) FIRED — investigate before next run");
    }
    eprintln!();

    // ── Hard assertions (test fails if critical thresholds violated) ──
    assert!(report.pct_worse <= ALERT_MAX_WORSE_PCT,
        "WORSE rate {:.1}% exceeds threshold {:.0}%",
        report.pct_worse * 100.0, ALERT_MAX_WORSE_PCT * 100.0);
    assert!(report.avg_top_k_overlap >= ALERT_MIN_AVG_OVERLAP,
        "avg overlap {:.3} below threshold {:.2}",
        report.avg_top_k_overlap, ALERT_MIN_AVG_OVERLAP);
    assert!(report.max_up_shift <= ALERT_MAX_POS_SHIFT,
        "max up shift {} exceeds cap {}", report.max_up_shift, ALERT_MAX_POS_SHIFT);
    assert!(report.max_down_shift <= ALERT_MAX_POS_SHIFT,
        "max down shift {} exceeds cap {}", report.max_down_shift, ALERT_MAX_POS_SHIFT);
    assert!(report.avg_latency_delta_us <= ALERT_MAX_AVG_LATENCY_US,
        "avg latency delta {:.0}μs exceeds {:.0}μs",
        report.avg_latency_delta_us, ALERT_MAX_AVG_LATENCY_US);
    assert!(report.pct_contradiction_worsened <= ALERT_MAX_CONTRADICTION_WORSENED_PCT,
        "contradiction worsened {:.1}% exceeds {:.0}%",
        report.pct_contradiction_worsened * 100.0, ALERT_MAX_CONTRADICTION_WORSENED_PCT * 100.0);
}
