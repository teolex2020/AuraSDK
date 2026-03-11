//! Candidate C Prerequisite: Concept Coverage Evaluation.
//!
//! Measures whether concepts have sufficient coverage over recall results
//! to justify implementing concept-assisted recall grouping.
//!
//! This is NOT an implementation of Candidate C — it is the prerequisite
//! measurement that determines whether C is worth starting.
//!
//! Gate: coverage >= 30% of recall results must have concept membership.

use aura::{Aura, Level};
use std::collections::{HashMap, HashSet};

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
// Per-query metrics
// ─────────────────────────────────────────────────────────

struct QueryCoverage {
    query: String,
    category: String,
    recall_count: usize,
    records_with_concept: usize,
    coverage: f32,
    distinct_concepts_hit: usize,
    avg_cluster_size: f32,
    max_cluster_size: usize,
    min_cluster_size: usize,
}

struct AggregateConceptMetrics {
    total_queries: usize,
    queries_with_results: usize,
    total_concepts: usize,
    stable_concepts: usize,
    candidate_concepts: usize,
    avg_coverage: f32,
    median_coverage: f32,
    pct_queries_with_any_concept: f32,
    pct_queries_above_30pct: f32,
    global_avg_cluster_size: f32,
    global_median_cluster_size: f32,
    avg_concepts_per_query: f32,
    max_concepts_per_query: usize,
    avg_records_per_concept: f32,
    max_records_per_concept: usize,
}

// ─────────────────────────────────────────────────────────
// Collection
// ─────────────────────────────────────────────────────────

fn collect_coverage(
    aura: &Aura,
    queries: &[(&str, usize, &str)],
) -> (Vec<QueryCoverage>, AggregateConceptMetrics) {
    // Build concept membership index: record_id → set of concept_ids
    let all_concepts = aura.get_concepts(None);
    let stable_concepts = aura.get_concepts(Some("stable"));
    let candidate_concepts = aura.get_concepts(Some("candidate"));

    let mut record_to_concepts: HashMap<String, Vec<String>> = HashMap::new();
    for concept in &all_concepts {
        for rid in &concept.record_ids {
            record_to_concepts.entry(rid.clone()).or_default().push(concept.id.clone());
        }
    }

    // Cluster sizes
    let cluster_sizes: Vec<usize> = all_concepts.iter()
        .filter(|c| !c.record_ids.is_empty())
        .map(|c| c.record_ids.len())
        .collect();

    let per_concept_records: Vec<usize> = all_concepts.iter()
        .map(|c| c.record_ids.len())
        .collect();

    // Per-query coverage
    let mut query_coverages: Vec<QueryCoverage> = Vec::new();

    for (query, top_k, category) in queries {
        let results = aura
            .recall_structured(query, Some(*top_k), Some(0.0), Some(true), None, None)
            .unwrap_or_default();

        let recall_count = results.len();
        if recall_count == 0 {
            query_coverages.push(QueryCoverage {
                query: query.to_string(),
                category: category.to_string(),
                recall_count: 0,
                records_with_concept: 0,
                coverage: 0.0,
                distinct_concepts_hit: 0,
                avg_cluster_size: 0.0,
                max_cluster_size: 0,
                min_cluster_size: 0,
            });
            continue;
        }

        let mut with_concept = 0usize;
        let mut concepts_hit: HashSet<String> = HashSet::new();
        let mut hit_cluster_sizes: Vec<usize> = Vec::new();

        for (_, rec) in &results {
            if let Some(cids) = record_to_concepts.get(&rec.id) {
                with_concept += 1;
                for cid in cids {
                    concepts_hit.insert(cid.clone());
                    if let Some(concept) = all_concepts.iter().find(|c| c.id == *cid) {
                        hit_cluster_sizes.push(concept.record_ids.len());
                    }
                }
            }
        }

        let coverage = with_concept as f32 / recall_count as f32;
        let avg_cluster = if !hit_cluster_sizes.is_empty() {
            hit_cluster_sizes.iter().sum::<usize>() as f32 / hit_cluster_sizes.len() as f32
        } else {
            0.0
        };
        let max_cluster = hit_cluster_sizes.iter().max().copied().unwrap_or(0);
        let min_cluster = hit_cluster_sizes.iter().min().copied().unwrap_or(0);

        query_coverages.push(QueryCoverage {
            query: query.to_string(),
            category: category.to_string(),
            recall_count,
            records_with_concept: with_concept,
            coverage,
            distinct_concepts_hit: concepts_hit.len(),
            avg_cluster_size: avg_cluster,
            max_cluster_size: max_cluster,
            min_cluster_size: min_cluster,
        });
    }

    // Aggregates
    let with_results: Vec<&QueryCoverage> = query_coverages.iter()
        .filter(|q| q.recall_count > 0)
        .collect();
    let n = with_results.len();

    let mut coverages: Vec<f32> = with_results.iter().map(|q| q.coverage).collect();
    coverages.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_coverage = if n > 0 { coverages.iter().sum::<f32>() / n as f32 } else { 0.0 };
    let median_coverage = if n > 0 {
        if n % 2 == 0 { (coverages[n/2 - 1] + coverages[n/2]) / 2.0 } else { coverages[n/2] }
    } else { 0.0 };

    let with_any_concept = with_results.iter().filter(|q| q.records_with_concept > 0).count();
    let above_30pct = with_results.iter().filter(|q| q.coverage >= 0.30).count();

    let avg_concepts_per_query = if n > 0 {
        with_results.iter().map(|q| q.distinct_concepts_hit as f32).sum::<f32>() / n as f32
    } else { 0.0 };
    let max_concepts_per_query = with_results.iter().map(|q| q.distinct_concepts_hit).max().unwrap_or(0);

    let mut sorted_clusters = cluster_sizes.clone();
    sorted_clusters.sort();
    let global_avg_cluster = if !sorted_clusters.is_empty() {
        sorted_clusters.iter().sum::<usize>() as f32 / sorted_clusters.len() as f32
    } else { 0.0 };
    let global_median_cluster = if !sorted_clusters.is_empty() {
        let m = sorted_clusters.len();
        if m % 2 == 0 {
            (sorted_clusters[m/2 - 1] + sorted_clusters[m/2]) as f32 / 2.0
        } else {
            sorted_clusters[m/2] as f32
        }
    } else { 0.0 };

    let avg_records_per_concept = if !per_concept_records.is_empty() {
        per_concept_records.iter().sum::<usize>() as f32 / per_concept_records.len() as f32
    } else { 0.0 };
    let max_records_per_concept = per_concept_records.iter().max().copied().unwrap_or(0);

    let agg = AggregateConceptMetrics {
        total_queries: queries.len(),
        queries_with_results: n,
        total_concepts: all_concepts.len(),
        stable_concepts: stable_concepts.len(),
        candidate_concepts: candidate_concepts.len(),
        avg_coverage,
        median_coverage,
        pct_queries_with_any_concept: if n > 0 { with_any_concept as f32 / n as f32 } else { 0.0 },
        pct_queries_above_30pct: if n > 0 { above_30pct as f32 / n as f32 } else { 0.0 },
        global_avg_cluster_size: global_avg_cluster,
        global_median_cluster_size: global_median_cluster,
        avg_concepts_per_query,
        max_concepts_per_query,
        avg_records_per_concept,
        max_records_per_concept,
    };

    (query_coverages, agg)
}

// ─────────────────────────────────────────────────────────
// Corpus (same as Phase 4 evidence for comparability)
// ─────────────────────────────────────────────────────────

fn build_corpus(aura: &Aura) {
    // Domain 1: Rust programming
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

    // Domain 2: Python
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

    // Domain 4: DevOps
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

    // Domain 5: Architecture
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

    // Domain 6: Database
    store_batch(aura, &[
        ("PostgreSQL is the default choice for relational data", Level::Domain, &["database", "postgresql", "relational"], "recorded", "preference"),
        ("Redis caching reduced API latency by seventy percent", Level::Domain, &["database", "redis", "caching", "performance"], "recorded", "fact"),
        ("MongoDB works well for document-oriented workloads", Level::Domain, &["database", "mongodb", "document"], "recorded", "fact"),
        ("Database migrations must be backward compatible for zero-downtime", Level::Domain, &["database", "migrations", "safety"], "recorded", "decision"),
        ("Connection pooling is essential for production database performance", Level::Domain, &["database", "performance", "pooling"], "recorded", "fact"),
        ("SQLite is ideal for embedded and local-first applications", Level::Domain, &["database", "sqlite", "embedded"], "recorded", "fact"),
        ("Database indexes should cover all frequent query patterns", Level::Domain, &["database", "indexes", "performance"], "recorded", "fact"),
    ]);

    // Domain 7: Workflow / identity
    store_batch(aura, &[
        ("I prefer dark mode in all development tools and editors", Level::Identity, &["preference", "editor", "dark-mode"], "recorded", "preference"),
        ("Neovim is my primary code editor for all projects", Level::Identity, &["preference", "editor", "neovim"], "recorded", "preference"),
        ("I review pull requests every morning before coding sessions", Level::Identity, &["workflow", "code-review", "morning"], "recorded", "preference"),
        ("Deep work blocks of two hours are most productive for me", Level::Identity, &["workflow", "productivity", "focus"], "recorded", "preference"),
        ("I always write tests before implementing features in TDD style", Level::Identity, &["workflow", "tdd", "testing"], "recorded", "preference"),
        ("I use pomodoro technique for focused coding sessions", Level::Identity, &["workflow", "pomodoro", "productivity"], "recorded", "preference"),
    ]);

    // Domain 8: Contested
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

    // Domain 9: Repeated facts
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

    // Domain 10: Security
    store_batch(aura, &[
        ("All API endpoints must validate input before processing", Level::Domain, &["security", "api", "validation"], "recorded", "decision"),
        ("Secrets must never be committed to version control", Level::Domain, &["security", "secrets", "git"], "recorded", "decision"),
        ("Dependencies should be audited regularly for vulnerabilities", Level::Domain, &["security", "dependencies", "audit"], "recorded", "decision"),
        ("HTTPS is mandatory for all production endpoints", Level::Domain, &["security", "https", "production"], "recorded", "decision"),
        ("Authentication tokens must have short expiry times", Level::Domain, &["security", "auth", "tokens"], "recorded", "decision"),
    ]);
}

// ─────────────────────────────────────────────────────────
// Query set (same as Phase 4 for comparability)
// ─────────────────────────────────────────────────────────

fn query_set() -> Vec<(&'static str, usize, &'static str)> {
    vec![
        // Stable factual
        ("Rust programming language", 10, "stable-factual"),
        ("Rust memory safety borrow checker", 10, "stable-factual"),
        ("Python data science numpy pandas", 10, "stable-factual"),
        ("TypeScript type safety web", 10, "stable-factual"),

        // Belief-heavy
        ("best language for backend systems", 10, "belief-heavy"),
        ("programming language choice for new project", 20, "belief-heavy"),
        ("deployment strategy for production", 20, "belief-heavy"),
        ("database choice relational vs document", 10, "belief-heavy"),
        ("architecture pattern for microservices", 20, "belief-heavy"),

        // Conflicting
        ("tabs vs spaces indentation", 10, "conflicting"),
        ("ORM vs raw SQL performance", 10, "conflicting"),
        ("monorepo vs polyrepo organization", 10, "conflicting"),
        ("GraphQL vs REST API design", 10, "conflicting"),

        // DevOps
        ("canary deployment production", 10, "devops"),
        ("feature flags deployment release", 10, "devops"),
        ("CI CD pipeline testing automation", 10, "devops"),
        ("infrastructure as code terraform", 10, "devops"),
        ("kubernetes pod self-healing", 10, "devops"),

        // Architecture
        ("microservices independent scaling", 10, "architecture"),
        ("event sourcing audit trail", 10, "architecture"),
        ("CQRS read write model", 10, "architecture"),
        ("circuit breaker cascade failure", 10, "architecture"),

        // Database
        ("PostgreSQL relational data", 10, "database"),
        ("Redis caching API latency", 10, "database"),
        ("database migrations backward compatible", 10, "database"),
        ("connection pooling production", 5, "database"),

        // Workflow
        ("dark mode development tools", 10, "workflow"),
        ("test driven development TDD", 10, "workflow"),
        ("deep work productivity focus", 10, "workflow"),

        // Security
        ("API input validation security", 10, "security"),
        ("secrets version control git", 10, "security"),
        ("authentication tokens expiry", 10, "security"),

        // Cross-domain
        ("testing deployment pipeline safety", 10, "cross-domain"),
        ("production safety monitoring strategy", 10, "cross-domain"),
        ("backend service architecture performance", 10, "cross-domain"),
        ("security audit deployment pipeline", 10, "cross-domain"),

        // Repeated facts
        ("trunk-based development git workflow", 10, "repeated"),
        ("team git branching strategy", 10, "repeated"),

        // Broad
        ("best practices software engineering", 20, "broad"),
        ("how to deploy safely to production", 20, "broad"),

        // No-match (edge cases)
        ("quantum computing blockchain integration", 10, "no-match"),
        ("recipe for chocolate cake baking", 10, "no-match"),
    ]
}

// ═════════════════════════════════════════════════════════
// CONCEPT COVERAGE EVALUATION TEST
// ═════════════════════════════════════════════════════════

#[test]
fn concept_coverage_evaluation() {
    let (aura, _dir) = open_temp_aura();

    // Build corpus and mature
    build_corpus(&aura);
    run_cycles(&aura, 12);

    // Collect coverage
    let queries = query_set();
    let (per_query, agg) = collect_coverage(&aura, &queries);

    // ── Report ──
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("CANDIDATE C PREREQUISITE: CONCEPT COVERAGE EVALUATION");
    eprintln!("{}", "=".repeat(80));
    eprintln!();
    eprintln!("── Concept Inventory ──");
    eprintln!("  Total concepts:       {}", agg.total_concepts);
    eprintln!("  Stable concepts:      {}", agg.stable_concepts);
    eprintln!("  Candidate concepts:   {}", agg.candidate_concepts);
    eprintln!("  Avg records/concept:  {:.1}", agg.avg_records_per_concept);
    eprintln!("  Max records/concept:  {}", agg.max_records_per_concept);
    eprintln!();
    eprintln!("── Coverage Over Recall Results ──");
    eprintln!("  Total queries:        {}", agg.total_queries);
    eprintln!("  With results:         {}", agg.queries_with_results);
    eprintln!("  Avg coverage:         {:.1}%", agg.avg_coverage * 100.0);
    eprintln!("  Median coverage:      {:.1}%", agg.median_coverage * 100.0);
    eprintln!("  % with any concept:   {:.1}%", agg.pct_queries_with_any_concept * 100.0);
    eprintln!("  % above 30% gate:     {:.1}%", agg.pct_queries_above_30pct * 100.0);
    eprintln!();
    eprintln!("── Cluster Metrics ──");
    eprintln!("  Avg cluster size:     {:.1}", agg.global_avg_cluster_size);
    eprintln!("  Median cluster size:  {:.1}", agg.global_median_cluster_size);
    eprintln!("  Avg concepts/query:   {:.1}", agg.avg_concepts_per_query);
    eprintln!("  Max concepts/query:   {}", agg.max_concepts_per_query);
    eprintln!();

    // Per-category breakdown
    let categories = [
        "stable-factual", "belief-heavy", "conflicting", "devops",
        "architecture", "database", "workflow", "security",
        "cross-domain", "repeated", "broad", "no-match",
    ];
    eprintln!("── Per-Category Coverage ──");
    for cat in &categories {
        let cat_queries: Vec<&QueryCoverage> = per_query.iter()
            .filter(|q| q.category == *cat && q.recall_count > 0)
            .collect();
        if cat_queries.is_empty() { continue; }

        let n = cat_queries.len();
        let avg_cov = cat_queries.iter().map(|q| q.coverage).sum::<f32>() / n as f32;
        let with_concept = cat_queries.iter().filter(|q| q.records_with_concept > 0).count();
        let avg_concepts = cat_queries.iter().map(|q| q.distinct_concepts_hit as f32).sum::<f32>() / n as f32;

        eprintln!(
            "  {:<16} n={:>2}  avg_cov={:>5.1}%  with_concept={:>2}/{:>2}  avg_concepts={:.1}",
            cat, n, avg_cov * 100.0, with_concept, n, avg_concepts,
        );
    }
    eprintln!();

    // Per-query detail for queries with concept membership
    let mut covered: Vec<&QueryCoverage> = per_query.iter()
        .filter(|q| q.records_with_concept > 0)
        .collect();
    covered.sort_by(|a, b| b.coverage.partial_cmp(&a.coverage).unwrap());

    if !covered.is_empty() {
        eprintln!("── Queries with Concept Membership ({}) ──", covered.len());
        for q in covered.iter().take(20) {
            eprintln!(
                "  [{:>5.1}% cov] q=\"{}\" cat={} results={} with_concept={} concepts={}  avg_cluster={:.1}",
                q.coverage * 100.0, q.query, q.category, q.recall_count,
                q.records_with_concept, q.distinct_concepts_hit, q.avg_cluster_size,
            );
        }
        if covered.len() > 20 {
            eprintln!("  ... and {} more", covered.len() - 20);
        }
        eprintln!();
    }

    // ── Gate Assessment ──
    eprintln!("{}", "=".repeat(80));
    eprintln!("CANDIDATE C GATE ASSESSMENT");
    eprintln!("{}", "=".repeat(80));

    // Gate 1: Coverage >= 30%
    let coverage_gate = agg.avg_coverage >= 0.30;
    eprintln!("  [{}] Coverage >= 30%: avg={:.1}%",
        if coverage_gate { "PASS" } else { "FAIL" }, agg.avg_coverage * 100.0);

    // Gate 2: Cluster granularity 3-15 records
    let granularity_gate = agg.global_avg_cluster_size >= 3.0 && agg.global_avg_cluster_size <= 15.0;
    eprintln!("  [{}] Cluster size 3-15: avg={:.1}",
        if granularity_gate { "PASS" } else { "FAIL" }, agg.global_avg_cluster_size);

    // Gate 3: At least some queries benefit
    let utility_gate = agg.pct_queries_with_any_concept >= 0.20;
    eprintln!("  [{}] >= 20% queries have concept signal: {:.1}%",
        if utility_gate { "PASS" } else { "FAIL" }, agg.pct_queries_with_any_concept * 100.0);

    // Gate 4: Multiple distinct concepts available
    let diversity_gate = agg.total_concepts >= 3;
    eprintln!("  [{}] >= 3 total concepts: {}",
        if diversity_gate { "PASS" } else { "FAIL" }, agg.total_concepts);

    eprintln!();

    let all_gates = coverage_gate && granularity_gate && utility_gate && diversity_gate;

    let verdict = if all_gates {
        "PROCEED — all prerequisites met, Candidate C implementation is justified"
    } else if utility_gate && diversity_gate {
        "DEFER — concepts exist but coverage too low. Wait for corpus maturation."
    } else if agg.total_concepts > 0 {
        "DEFER — concept layer is active but insufficient for recall grouping."
    } else {
        "DEFER — no concepts formed. Concept layer needs more data or cycles."
    };

    eprintln!("VERDICT: {}", verdict);
    eprintln!();

    // ── Informational assertions (not hard failures) ──
    // We don't assert the gates because C is expected to be deferred.
    // We only assert the test infrastructure works.
    assert!(agg.total_queries >= 40,
        "need >= 40 queries, got {}", agg.total_queries);
    assert!(agg.queries_with_results >= 30,
        "need >= 30 queries with results, got {}", agg.queries_with_results);
}

// ═════════════════════════════════════════════════════════
// WIDER RE-EVALUATION: partition diagnostics + enriched corpus
// ═════════════════════════════════════════════════════════

/// Diagnostic: show belief partition structure to understand why concepts don't form.
#[test]
fn concept_partition_diagnostic() {
    let (aura, _dir) = open_temp_aura();
    build_corpus(&aura);
    run_cycles(&aura, 15);

    let beliefs = aura.get_beliefs(None);

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("PARTITION DIAGNOSTIC — standard corpus, 15 cycles");
    eprintln!("{}", "=".repeat(80));
    eprintln!("  Total beliefs: {}", beliefs.len());

    // Extract partition key from belief key: first segment = namespace, last = semantic_type
    let mut partitions: HashMap<String, Vec<&aura::belief::Belief>> = HashMap::new();
    for b in &beliefs {
        let parts: Vec<&str> = b.key.split(':').collect();
        let ns = parts.first().copied().unwrap_or("?");
        let st = parts.last().copied().unwrap_or("?");
        let partition_key = format!("{}:{}", ns, st);
        partitions.entry(partition_key).or_default().push(b);
    }

    let mut keys: Vec<&String> = partitions.keys().collect();
    keys.sort();

    for pk in &keys {
        let beliefs_in_part = &partitions[*pk];
        let stable_count = beliefs_in_part.iter()
            .filter(|b| b.stability >= 2.0 && b.confidence >= 0.55)
            .count();
        eprintln!("\n  Partition [{}] — {} beliefs, {} pass seed gate",
            pk, beliefs_in_part.len(), stable_count);
        for b in beliefs_in_part {
            let passes = b.stability >= 2.0 && b.confidence >= 0.55;
            eprintln!("    {} key={} stab={:.1} conf={:.2} state={:?}",
                if passes { "✓" } else { "✗" },
                b.key, b.stability, b.confidence, b.state);
        }
    }

    // Summary
    let multi_seed_partitions = partitions.iter()
        .filter(|(_, bs)| {
            bs.iter().filter(|b| b.stability >= 2.0 && b.confidence >= 0.55).count() >= 2
        })
        .count();
    eprintln!("\n  Partitions with >= 2 seeds: {} (need >= 1 for any concept)", multi_seed_partitions);
    eprintln!();
}

/// Enriched corpus: adds paraphrases within same semantic_type to boost partition density.
fn build_enriched_corpus(aura: &Aura) {
    // Start with the standard corpus
    build_corpus(aura);

    // Add paraphrases to boost per-partition density
    // These records share (namespace=default, semantic_type) with existing records
    // to ensure multiple beliefs land in the same partition.

    // Boost "decision" partition — most split across domains
    store_batch(aura, &[
        // DevOps decisions (match existing deploy decisions)
        ("Production deployments require staged rollout approval", Level::Domain, &["deploy", "production", "safety"], "recorded", "decision"),
        ("All releases must pass canary validation before full deploy", Level::Domain, &["deploy", "canary", "validation"], "recorded", "decision"),
        ("Rollback plans are required for every production release", Level::Domain, &["deploy", "rollback", "safety"], "recorded", "decision"),
        // Security decisions (match existing security decisions)
        ("Rate limiting must be applied to all public API endpoints", Level::Domain, &["security", "api", "rate-limit"], "recorded", "decision"),
        ("All database access must go through parameterized queries", Level::Domain, &["security", "database", "sql-injection"], "recorded", "decision"),
        // Database decisions (match existing)
        ("Database schema changes require a migration review", Level::Domain, &["database", "migrations", "review"], "recorded", "decision"),
        ("All production queries must use connection pooling", Level::Domain, &["database", "pooling", "production"], "recorded", "decision"),
    ]);

    // Boost "fact" partition — many facts across domains
    store_batch(aura, &[
        // More Rust facts
        ("Rust lifetime system ensures references are always valid", Level::Domain, &["rust", "programming", "lifetimes"], "recorded", "fact"),
        ("Rust pattern matching is exhaustive and compiler-checked", Level::Domain, &["rust", "programming", "patterns"], "recorded", "fact"),
        // More Python facts
        ("Python asyncio enables concurrent IO-bound operations", Level::Domain, &["python", "programming", "async"], "recorded", "fact"),
        ("Python virtual environments isolate project dependencies", Level::Domain, &["python", "programming", "venv"], "recorded", "fact"),
        // More DevOps facts
        ("Prometheus collects metrics for infrastructure monitoring", Level::Domain, &["monitoring", "prometheus", "ops"], "recorded", "fact"),
        ("Grafana dashboards visualize infrastructure health metrics", Level::Domain, &["monitoring", "grafana", "ops"], "recorded", "fact"),
        // More Architecture facts
        ("Service mesh handles inter-service communication and observability", Level::Domain, &["architecture", "service-mesh", "observability"], "recorded", "fact"),
        ("Strangler fig pattern enables incremental migration from monolith", Level::Domain, &["architecture", "strangler-fig", "migration"], "recorded", "fact"),
    ]);

    // Boost "preference" partition
    store_batch(aura, &[
        ("I prefer writing documentation alongside code changes", Level::Identity, &["workflow", "documentation", "practice"], "recorded", "preference"),
        ("Pair programming is my preferred approach for complex features", Level::Identity, &["workflow", "pair-programming", "collaboration"], "recorded", "preference"),
        ("I always use a linter for consistent code style enforcement", Level::Identity, &["workflow", "linting", "style"], "recorded", "preference"),
        ("Go is my secondary language for CLI tool development", Level::Domain, &["go", "programming", "cli"], "recorded", "preference"),
    ]);
}

/// Extended query set — includes queries targeting enriched content.
fn extended_query_set() -> Vec<(&'static str, usize, &'static str)> {
    let mut queries = query_set();
    queries.extend(vec![
        // Targeting enriched decision content
        ("staged rollout production deployment", 10, "enriched-decision"),
        ("canary validation deployment strategy", 10, "enriched-decision"),
        ("database migration review process", 10, "enriched-decision"),
        ("rate limiting API security", 10, "enriched-decision"),
        // Targeting enriched fact content
        ("Rust lifetime borrow system", 10, "enriched-fact"),
        ("Python asyncio concurrent programming", 10, "enriched-fact"),
        ("Prometheus Grafana monitoring", 10, "enriched-fact"),
        ("service mesh observability", 10, "enriched-fact"),
        // Targeting enriched preferences
        ("pair programming collaboration", 10, "enriched-preference"),
        ("code documentation writing practice", 10, "enriched-preference"),
    ]);
    queries
}

/// Wider re-evaluation with enriched corpus, more cycles, and diagnostics.
#[test]
fn concept_coverage_wider_reeval() {
    let (aura, _dir) = open_temp_aura();

    // Build enriched corpus and mature longer
    build_enriched_corpus(&aura);
    run_cycles(&aura, 20);

    // Get maintenance report for concept diagnostics
    let report = aura.run_maintenance();
    {
        let cr = &report.concept;
        eprintln!("\n── Concept Phase Diagnostics (cycle 21) ──");
        eprintln!("  seeds_found:       {}", cr.seeds_found);
        eprintln!("  centroids_built:   {}", cr.centroids_built);
        eprintln!("  partitions≥2:      {}", cr.partitions_with_multiple_seeds);
        eprintln!("  pairwise:          {}", cr.pairwise_comparisons);
        eprintln!("  above_threshold:   {}", cr.pairwise_above_threshold);
        eprintln!("  tanimoto_min:      {:.4}", cr.tanimoto_min);
        eprintln!("  tanimoto_max:      {:.4}", cr.tanimoto_max);
        eprintln!("  tanimoto_avg:      {:.4}", cr.tanimoto_avg);
        eprintln!("  avg_centroid_size: {:.0}", cr.avg_centroid_size);
        eprintln!("  candidates_found:  {}", cr.candidates_found);
        eprintln!("  stable_count:      {}", cr.stable_count);
    }

    // Partition diagnostic
    let beliefs = aura.get_beliefs(None);
    let mut partitions: HashMap<String, Vec<&aura::belief::Belief>> = HashMap::new();
    for b in &beliefs {
        let parts: Vec<&str> = b.key.split(':').collect();
        let ns = parts.first().copied().unwrap_or("?");
        let st = parts.last().copied().unwrap_or("?");
        let partition_key = format!("{}:{}", ns, st);
        partitions.entry(partition_key).or_default().push(b);
    }

    eprintln!("\n── Belief Partitions (enriched corpus, 21 cycles) ──");
    let mut keys: Vec<&String> = partitions.keys().collect();
    keys.sort();
    for pk in &keys {
        let bs = &partitions[*pk];
        let seeds = bs.iter().filter(|b| b.stability >= 2.0 && b.confidence >= 0.55).count();
        eprintln!("  [{}] beliefs={} seeds={}", pk, bs.len(), seeds);
    }

    let multi_seed = partitions.iter()
        .filter(|(_, bs)| bs.iter().filter(|b| b.stability >= 2.0 && b.confidence >= 0.55).count() >= 2)
        .count();
    eprintln!("  Partitions with >= 2 seeds: {}", multi_seed);

    // Collect coverage
    let queries = extended_query_set();
    let (per_query, agg) = collect_coverage(&aura, &queries);

    // ── Report ──
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("CANDIDATE C WIDER RE-EVALUATION (enriched corpus, 21 cycles, {} queries)", queries.len());
    eprintln!("{}", "=".repeat(80));
    eprintln!();
    eprintln!("── Concept Inventory ──");
    eprintln!("  Total concepts:       {}", agg.total_concepts);
    eprintln!("  Stable concepts:      {}", agg.stable_concepts);
    eprintln!("  Candidate concepts:   {}", agg.candidate_concepts);
    eprintln!("  Avg records/concept:  {:.1}", agg.avg_records_per_concept);
    eprintln!("  Max records/concept:  {}", agg.max_records_per_concept);
    eprintln!();
    eprintln!("── Coverage Over Recall Results ──");
    eprintln!("  Total queries:        {}", agg.total_queries);
    eprintln!("  With results:         {}", agg.queries_with_results);
    eprintln!("  Avg coverage:         {:.1}%", agg.avg_coverage * 100.0);
    eprintln!("  Median coverage:      {:.1}%", agg.median_coverage * 100.0);
    eprintln!("  % with any concept:   {:.1}%", agg.pct_queries_with_any_concept * 100.0);
    eprintln!("  % above 30% gate:     {:.1}%", agg.pct_queries_above_30pct * 100.0);
    eprintln!();

    // Per-category breakdown
    let categories = [
        "stable-factual", "belief-heavy", "conflicting", "devops",
        "architecture", "database", "workflow", "security",
        "cross-domain", "repeated", "broad", "no-match",
        "enriched-decision", "enriched-fact", "enriched-preference",
    ];
    eprintln!("── Per-Category Coverage ──");
    for cat in &categories {
        let cat_queries: Vec<&QueryCoverage> = per_query.iter()
            .filter(|q| q.category == *cat && q.recall_count > 0)
            .collect();
        if cat_queries.is_empty() { continue; }
        let n = cat_queries.len();
        let avg_cov = cat_queries.iter().map(|q| q.coverage).sum::<f32>() / n as f32;
        let with_concept = cat_queries.iter().filter(|q| q.records_with_concept > 0).count();
        eprintln!("  {:<20} n={:>2}  avg_cov={:>5.1}%  with_concept={:>2}/{:>2}",
            cat, n, avg_cov * 100.0, with_concept, n);
    }
    eprintln!();

    // Queries with concept membership
    let mut covered: Vec<&QueryCoverage> = per_query.iter()
        .filter(|q| q.records_with_concept > 0)
        .collect();
    covered.sort_by(|a, b| b.coverage.partial_cmp(&a.coverage).unwrap());
    if !covered.is_empty() {
        eprintln!("── Queries with Concept Membership ({}) ──", covered.len());
        for q in covered.iter().take(20) {
            eprintln!("  [{:>5.1}% cov] q=\"{}\" cat={} results={} with_concept={} concepts={}",
                q.coverage * 100.0, q.query, q.category, q.recall_count,
                q.records_with_concept, q.distinct_concepts_hit);
        }
        eprintln!();
    }

    // ── Gate Assessment ──
    eprintln!("{}", "=".repeat(80));
    eprintln!("CANDIDATE C WIDER RE-EVALUATION GATES");
    eprintln!("{}", "=".repeat(80));
    let coverage_gate = agg.avg_coverage >= 0.30;
    let utility_gate = agg.pct_queries_with_any_concept >= 0.20;
    let diversity_gate = agg.total_concepts >= 3;
    let concepts_exist = agg.total_concepts > 0;
    eprintln!("  [{}] Coverage >= 30%: {:.1}%",
        if coverage_gate { "PASS" } else { "FAIL" }, agg.avg_coverage * 100.0);
    eprintln!("  [{}] >= 20% queries with concept signal: {:.1}%",
        if utility_gate { "PASS" } else { "FAIL" }, agg.pct_queries_with_any_concept * 100.0);
    eprintln!("  [{}] >= 3 total concepts: {}",
        if diversity_gate { "PASS" } else { "FAIL" }, agg.total_concepts);
    eprintln!("  [{}] Any concepts formed: {}",
        if concepts_exist { "PASS" } else { "FAIL" }, agg.total_concepts);
    eprintln!();

    let verdict = if coverage_gate && utility_gate && diversity_gate {
        "PROCEED — all gates pass, Candidate C ready for inspection-only grouping"
    } else if concepts_exist {
        "PARTIAL SIGNAL — concepts form but coverage insufficient for promotion"
    } else {
        "STRUCTURAL LIMIT — 0 concepts; partition granularity or corpus diversity blocks formation"
    };
    eprintln!("VERDICT: {}", verdict);
    eprintln!();

    // Infrastructure assertions
    assert!(agg.total_queries >= 50, "need >= 50 queries, got {}", agg.total_queries);
    assert!(agg.queries_with_results >= 40, "need >= 40 queries with results, got {}", agg.queries_with_results);
}
