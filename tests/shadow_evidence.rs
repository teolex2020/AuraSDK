//! Phase 3 Step 3: Shadow evidence collection.
//!
//! Runs >= 50 diverse recall queries through `recall_structured_with_shadow()`,
//! collects per-query `ShadowRecallReport` metrics, computes aggregate statistics,
//! and validates review thresholds.
//!
//! This is the evidence artifact for the Candidate B promotion decision.

use aura::{Aura, Level};

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

/// Per-query evidence record.
struct QueryEvidence {
    query: String,
    top_k: usize,
    baseline_count: usize,
    top_k_overlap: f32,
    promoted_count: usize,
    demoted_count: usize,
    unchanged_count: usize,
    belief_coverage: f32,
    avg_belief_multiplier: f32,
    shadow_latency_us: u64,
    has_movement: bool,
}

/// Aggregate statistics from evidence collection.
struct AggregateMetrics {
    total_queries: usize,
    queries_with_results: usize,
    avg_top_k_overlap: f32,
    median_top_k_overlap: f32,
    pct_with_beneficial_movement: f32,
    pct_with_any_movement: f32,
    pct_with_large_divergence: f32,
    avg_belief_coverage: f32,
    avg_latency_us: f64,
    max_latency_us: u64,
    p95_latency_us: u64,
}

fn collect_evidence(aura: &Aura, queries: &[(&str, usize)]) -> Vec<QueryEvidence> {
    queries.iter().map(|(query, top_k)| {
        let (baseline, shadow) = aura
            .recall_structured_with_shadow(
                query,
                Some(*top_k),
                Some(0.0),
                Some(true),
                None,
                None,
            )
            .expect("shadow recall failed");

        let has_movement = shadow.promoted_count > 0 || shadow.demoted_count > 0;

        QueryEvidence {
            query: query.to_string(),
            top_k: *top_k,
            baseline_count: baseline.len(),
            top_k_overlap: shadow.top_k_overlap,
            promoted_count: shadow.promoted_count,
            demoted_count: shadow.demoted_count,
            unchanged_count: shadow.unchanged_count,
            belief_coverage: shadow.belief_coverage,
            avg_belief_multiplier: shadow.avg_belief_multiplier,
            shadow_latency_us: shadow.shadow_latency_us,
            has_movement,
        }
    }).collect()
}

fn compute_aggregates(evidence: &[QueryEvidence]) -> AggregateMetrics {
    let total = evidence.len();
    let with_results: Vec<&QueryEvidence> = evidence.iter()
        .filter(|e| e.baseline_count > 0)
        .collect();
    let n = with_results.len();

    // Overlap stats (only for queries that returned results)
    let mut overlaps: Vec<f32> = with_results.iter().map(|e| e.top_k_overlap).collect();
    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_overlap = if n > 0 {
        overlaps.iter().sum::<f32>() / n as f32
    } else {
        1.0
    };
    let median_overlap = if n > 0 {
        if n % 2 == 0 {
            (overlaps[n / 2 - 1] + overlaps[n / 2]) / 2.0
        } else {
            overlaps[n / 2]
        }
    } else {
        1.0
    };

    // Movement stats
    let with_movement = with_results.iter().filter(|e| e.has_movement).count();
    let with_beneficial = with_results.iter()
        .filter(|e| e.promoted_count > 0)
        .count();
    let with_large_divergence = with_results.iter()
        .filter(|e| e.top_k_overlap < 0.5)
        .count();

    // Coverage
    let avg_coverage = if n > 0 {
        with_results.iter().map(|e| e.belief_coverage).sum::<f32>() / n as f32
    } else {
        0.0
    };

    // Latency
    let mut latencies: Vec<u64> = with_results.iter().map(|e| e.shadow_latency_us).collect();
    latencies.sort();
    let avg_latency = if n > 0 {
        latencies.iter().sum::<u64>() as f64 / n as f64
    } else {
        0.0
    };
    let max_latency = latencies.last().copied().unwrap_or(0);
    let p95_latency = if n > 0 {
        latencies[((n as f64 * 0.95) as usize).min(n - 1)]
    } else {
        0
    };

    AggregateMetrics {
        total_queries: total,
        queries_with_results: n,
        avg_top_k_overlap: avg_overlap,
        median_top_k_overlap: median_overlap,
        pct_with_beneficial_movement: if n > 0 { with_beneficial as f32 / n as f32 } else { 0.0 },
        pct_with_any_movement: if n > 0 { with_movement as f32 / n as f32 } else { 0.0 },
        pct_with_large_divergence: if n > 0 { with_large_divergence as f32 / n as f32 } else { 0.0 },
        avg_belief_coverage: avg_coverage,
        avg_latency_us: avg_latency,
        max_latency_us: max_latency,
        p95_latency_us: p95_latency,
    }
}

// ─────────────────────────────────────────────────────────
// Corpus: diverse records across multiple domains
// ─────────────────────────────────────────────────────────

fn build_diverse_corpus(aura: &Aura) {
    // Domain 1: Programming preferences (should form beliefs)
    store_batch(aura, &[
        ("Rust is the best language for systems programming", Level::Domain, &["rust", "programming", "systems"], "recorded", "preference"),
        ("Rust memory safety prevents buffer overflows", Level::Domain, &["rust", "programming", "safety"], "recorded", "fact"),
        ("Rust borrow checker catches data races at compile time", Level::Domain, &["rust", "programming", "safety"], "recorded", "fact"),
        ("I always choose Rust for new backend services", Level::Domain, &["rust", "programming", "backend"], "recorded", "preference"),
        ("Rust async ecosystem has matured significantly", Level::Domain, &["rust", "programming", "async"], "recorded", "fact"),
        ("Python is easier to learn than Rust", Level::Domain, &["python", "programming", "learning"], "recorded", "fact"),
        ("Python data science ecosystem is unmatched", Level::Domain, &["python", "programming", "data"], "recorded", "fact"),
        ("I use Python for quick prototyping and scripting", Level::Domain, &["python", "programming", "scripting"], "recorded", "preference"),
        ("TypeScript adds type safety to JavaScript projects", Level::Domain, &["typescript", "programming", "web"], "recorded", "fact"),
        ("TypeScript catches errors before runtime", Level::Domain, &["typescript", "programming", "safety"], "recorded", "fact"),
    ]);

    // Domain 2: DevOps decisions (should form causal patterns → beliefs)
    store_batch(aura, &[
        ("Deployed canary releases to production environment", Level::Domain, &["deploy", "canary", "ops"], "recorded", "decision"),
        ("Canary deployment caught critical regression early", Level::Domain, &["deploy", "canary", "ops"], "recorded", "fact"),
        ("Blue-green deployment reduces downtime during releases", Level::Domain, &["deploy", "blue-green", "ops"], "recorded", "fact"),
        ("Feature flags decouple deployment from release", Level::Domain, &["deploy", "feature-flags", "ops"], "recorded", "decision"),
        ("CI/CD pipeline runs all tests before deployment", Level::Domain, &["deploy", "ci-cd", "testing"], "recorded", "fact"),
        ("Rollback procedure must be tested before each release", Level::Domain, &["deploy", "rollback", "ops"], "recorded", "decision"),
        ("Monitoring alerts should fire within two minutes of anomaly", Level::Domain, &["monitoring", "alerting", "ops"], "recorded", "fact"),
        ("Infrastructure as code prevents configuration drift", Level::Domain, &["infra", "iac", "ops"], "recorded", "fact"),
    ]);

    // Domain 3: Architecture patterns
    store_batch(aura, &[
        ("Microservices enable independent team scaling", Level::Domain, &["architecture", "microservices"], "recorded", "fact"),
        ("Event sourcing provides complete audit trail", Level::Domain, &["architecture", "event-sourcing"], "recorded", "fact"),
        ("CQRS separates read and write models for scalability", Level::Domain, &["architecture", "cqrs"], "recorded", "fact"),
        ("Domain driven design aligns code with business concepts", Level::Domain, &["architecture", "ddd"], "recorded", "fact"),
        ("API gateway pattern simplifies client integration", Level::Domain, &["architecture", "api-gateway"], "recorded", "fact"),
        ("Circuit breaker pattern prevents cascade failures", Level::Domain, &["architecture", "resilience"], "recorded", "fact"),
    ]);

    // Domain 4: Database decisions
    store_batch(aura, &[
        ("PostgreSQL is the default choice for relational data", Level::Domain, &["database", "postgresql"], "recorded", "preference"),
        ("Redis caching reduced API latency by seventy percent", Level::Domain, &["database", "redis", "caching"], "recorded", "fact"),
        ("MongoDB works well for document-oriented workloads", Level::Domain, &["database", "mongodb"], "recorded", "fact"),
        ("Database migrations must be backward compatible", Level::Domain, &["database", "migrations"], "recorded", "decision"),
        ("Connection pooling is essential for production databases", Level::Domain, &["database", "performance"], "recorded", "fact"),
    ]);

    // Domain 5: Personal workflow
    store_batch(aura, &[
        ("I prefer dark mode in all development tools", Level::Identity, &["preference", "editor", "dark-mode"], "recorded", "preference"),
        ("Neovim is my primary code editor", Level::Identity, &["preference", "editor", "neovim"], "recorded", "preference"),
        ("I review pull requests every morning before coding", Level::Identity, &["workflow", "code-review"], "recorded", "preference"),
        ("Deep work blocks of two hours are most productive", Level::Identity, &["workflow", "productivity"], "recorded", "preference"),
        ("I always write tests before implementing features", Level::Identity, &["workflow", "tdd", "testing"], "recorded", "preference"),
    ]);

    // Domain 6: Contradictions / contested beliefs
    store_batch(aura, &[
        ("Tabs are better than spaces for indentation", Level::Working, &["style", "formatting"], "recorded", "preference"),
        ("Spaces are better than tabs for consistent alignment", Level::Working, &["style", "formatting"], "recorded", "preference"),
        ("ORM is necessary for database abstraction", Level::Working, &["database", "orm"], "recorded", "preference"),
        ("Raw SQL is better than ORM for complex queries", Level::Working, &["database", "orm", "sql"], "recorded", "preference"),
    ]);

    // Domain 7: Facts that repeat (should form strong singleton beliefs)
    for i in 0..6 {
        aura.store(
            &format!("The team uses trunk-based development workflow iteration {}", i),
            Some(Level::Domain),
            Some(vec!["workflow".to_string(), "git".to_string(), "trunk".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }
}

/// 50+ diverse queries covering all domains, various top_k values.
fn query_set() -> Vec<(&'static str, usize)> {
    vec![
        // Programming — direct hits
        ("Rust programming language", 10),
        ("Python data science", 10),
        ("TypeScript type safety", 10),
        ("Rust memory safety borrow checker", 10),
        ("Python scripting prototyping", 5),
        ("best language for backend", 10),
        ("programming language choice", 20),
        ("Rust async ecosystem", 5),
        ("type safety compile time", 10),
        ("systems programming", 10),
        // DevOps — direct hits
        ("canary deployment production", 10),
        ("blue green deployment", 10),
        ("feature flags deployment", 10),
        ("CI CD pipeline testing", 10),
        ("rollback procedure release", 5),
        ("monitoring alerting anomaly", 10),
        ("infrastructure as code", 10),
        ("deployment strategy", 20),
        // Architecture
        ("microservices scaling", 10),
        ("event sourcing audit", 10),
        ("CQRS read write", 10),
        ("domain driven design", 10),
        ("API gateway pattern", 5),
        ("circuit breaker resilience", 10),
        ("architecture patterns", 20),
        // Database
        ("PostgreSQL relational", 10),
        ("Redis caching performance", 10),
        ("MongoDB document", 10),
        ("database migrations", 10),
        ("connection pooling production", 5),
        ("database choice", 20),
        // Personal workflow
        ("dark mode editor", 10),
        ("neovim editor", 5),
        ("code review morning", 10),
        ("deep work productivity", 10),
        ("test driven development", 10),
        ("coding workflow preference", 20),
        // Cross-domain
        ("testing deployment pipeline", 10),
        ("production safety strategy", 10),
        ("backend service architecture", 10),
        ("performance optimization caching", 10),
        ("development tools preference", 10),
        // Contested topics
        ("tabs vs spaces", 10),
        ("ORM vs raw SQL", 10),
        ("indentation style", 10),
        // Repeated facts
        ("trunk based development", 10),
        ("git workflow team", 10),
        // Broad queries
        ("best practices", 20),
        ("how to deploy safely", 20),
        ("what programming language to use", 20),
        // Edge cases: novel queries (low expected overlap)
        ("quantum computing blockchain", 10),
        ("gardening tips for spring", 5),
        ("recipe for chocolate cake", 10),
    ]
}

// ═════════════════════════════════════════════════════════
// EVIDENCE COLLECTION TEST
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_evidence_collection() {
    let (aura, _dir) = open_temp_aura();

    // Build corpus
    build_diverse_corpus(&aura);

    // Run enough maintenance cycles for beliefs to form and stabilize
    run_cycles(&aura, 10);

    // Collect evidence
    let queries = query_set();
    assert!(queries.len() >= 50, "need >= 50 queries, got {}", queries.len());

    let evidence = collect_evidence(&aura, &queries);
    let agg = compute_aggregates(&evidence);

    // ── Print evidence report ──
    eprintln!("\n{}", "=".repeat(72));
    eprintln!("SHADOW EVIDENCE REPORT — {} queries", agg.total_queries);
    eprintln!("{}", "=".repeat(72));
    eprintln!("Queries with results:       {}/{}", agg.queries_with_results, agg.total_queries);
    eprintln!("Avg top-k overlap:          {:.3}", agg.avg_top_k_overlap);
    eprintln!("Median top-k overlap:       {:.3}", agg.median_top_k_overlap);
    eprintln!("% with any movement:        {:.1}%", agg.pct_with_any_movement * 100.0);
    eprintln!("% with beneficial movement: {:.1}%", agg.pct_with_beneficial_movement * 100.0);
    eprintln!("% with large divergence:    {:.1}%", agg.pct_with_large_divergence * 100.0);
    eprintln!("Avg belief coverage:        {:.3}", agg.avg_belief_coverage);
    eprintln!("Avg latency:                {:.0}μs", agg.avg_latency_us);
    eprintln!("P95 latency:                {}μs", agg.p95_latency_us);
    eprintln!("Max latency:                {}μs", agg.max_latency_us);
    eprintln!();

    // Per-query detail for queries with movement
    let mut moved: Vec<&QueryEvidence> = evidence.iter()
        .filter(|e| e.has_movement && e.baseline_count > 0)
        .collect();
    moved.sort_by(|a, b| a.top_k_overlap.partial_cmp(&b.top_k_overlap).unwrap());

    if !moved.is_empty() {
        eprintln!("Queries with shadow movement ({}):", moved.len());
        for e in &moved {
            eprintln!(
                "  [{:.2} overlap] q=\"{}\" top_k={} results={} +{}/-{} ={}  cov={:.2}  mul={:.3}",
                e.top_k_overlap, e.query, e.top_k, e.baseline_count,
                e.promoted_count, e.demoted_count, e.unchanged_count,
                e.belief_coverage, e.avg_belief_multiplier,
            );
        }
    } else {
        eprintln!("No queries showed shadow movement.");
    }
    eprintln!();

    // ── Threshold assertions ──

    // 1. Overlap must not be chaotic: avg overlap >= 0.70
    //    (shadow should mostly agree with baseline)
    assert!(agg.avg_top_k_overlap >= 0.70,
        "avg overlap {:.3} below 0.70 — shadow scoring is too divergent", agg.avg_top_k_overlap);

    // 2. No extreme divergence: < 10% of queries with overlap < 0.5
    assert!(agg.pct_with_large_divergence < 0.10,
        "{}% queries have overlap < 0.5 — exceeds 10% threshold",
        (agg.pct_with_large_divergence * 100.0) as u32);

    // 3. Latency budget: P95 < 2ms (2000μs)
    assert!(agg.p95_latency_us < 2000,
        "P95 latency {}μs exceeds 2ms budget", agg.p95_latency_us);

    // 4. Max latency: no query > 5ms
    assert!(agg.max_latency_us < 5000,
        "max latency {}μs exceeds 5ms", agg.max_latency_us);

    // 5. At least some queries must return results
    assert!(agg.queries_with_results >= 30,
        "only {} queries returned results — corpus too sparse", agg.queries_with_results);

    // ── Signal assessment (informational, not hard assertions) ──
    // These determine whether B has a meaningful signal.

    let signal_useful = agg.avg_belief_coverage > 0.05
        && agg.pct_with_any_movement > 0.0;

    eprintln!("Signal assessment:");
    eprintln!("  belief_coverage > 0.05:  {} (actual: {:.3})",
        if agg.avg_belief_coverage > 0.05 { "YES" } else { "NO" }, agg.avg_belief_coverage);
    eprintln!("  any movement > 0%:       {} (actual: {:.1}%)",
        if agg.pct_with_any_movement > 0.0 { "YES" } else { "NO" }, agg.pct_with_any_movement * 100.0);
    eprintln!("  beneficial movement > 0%: {} (actual: {:.1}%)",
        if agg.pct_with_beneficial_movement > 0.0 { "YES" } else { "NO" }, agg.pct_with_beneficial_movement * 100.0);
    eprintln!("  Signal useful: {}", if signal_useful { "YES" } else { "NO — shadow path shows no meaningful effect" });
    eprintln!();

    // ── Verdict ──
    let verdict = if !signal_useful {
        "DEFER — shadow scoring shows no meaningful signal yet"
    } else if agg.avg_top_k_overlap >= 0.90 && agg.pct_with_beneficial_movement > 0.10 {
        "PROMOTE — shadow scoring safe and shows beneficial signal"
    } else if agg.avg_top_k_overlap >= 0.70 {
        "CONTINUE SHADOW — safe but needs more data"
    } else {
        "REJECT — too divergent from baseline"
    };

    eprintln!("VERDICT: {}", verdict);
    eprintln!();
}
