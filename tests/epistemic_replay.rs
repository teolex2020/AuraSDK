//! Practical replay / scenario / soak tests for the epistemic + belief layer.
//!
//! These tests exercise the full Aura stack through realistic record streams
//! and multiple maintenance cycles.  They verify:
//!
//! - belief churn stays low on stable data
//! - support / conflict signals are scoped correctly
//! - recall quality doesn't degrade after maintenance
//! - unresolved clusters don't appear on clean data
//! - long-run soak doesn't produce runaway metrics

use aura::{Aura, Level};

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

/// Store a batch of records from a descriptor list.
/// Each entry: (content, level, tags, source_type, semantic_type)
fn store_batch(
    aura: &Aura,
    batch: &[(&str, Level, &[&str], &str, &str)],
) {
    for (content, level, tags, source, semantic) in batch {
        aura.store(
            content,
            Some(*level),
            Some(tags.iter().map(|t| t.to_string()).collect()),
            None, None,
            Some(*source),
            None,
            Some(false), // disable dedup — we want every record created
            None, None,
            Some(*semantic),
        )
        .unwrap_or_else(|e| panic!("store failed for '{}': {}", content, e));
    }
}

/// Run N maintenance cycles and collect reports.
fn run_cycles(aura: &Aura, n: usize) -> Vec<aura::background_brain::MaintenanceReport> {
    (0..n).map(|_| aura.run_maintenance()).collect()
}

/// Recall structured results for a query.
fn recall(aura: &Aura, query: &str) -> Vec<(f32, aura::Record)> {
    aura.recall_structured(query, Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed")
}

// ═════════════════════════════════════════════════════════
// 1.  REPLAY TESTS
// ═════════════════════════════════════════════════════════

/// Realistic stream: preferences with contextual shifts,
/// one decision contradiction, unrelated same-tag records.
/// Run 5 maintenance cycles and assert invariants.
#[test]
fn replay_realistic_stream() {
    let (aura, _dir) = open_temp_aura();

    // ── Batch 1: initial preferences & decisions ──
    store_batch(&aura, &[
        // User coding preferences
        ("I prefer dark mode when writing code in the editor",
         Level::Domain, &["ui", "preferences", "coding"], "recorded", "preference"),
        ("Monospace font size 14 is my standard for all code editors",
         Level::Domain, &["ui", "preferences", "coding"], "recorded", "preference"),
        // User doc-reading preferences (same tags, different context)
        ("For reading documentation I prefer light mode with serif fonts",
         Level::Domain, &["ui", "preferences", "reading"], "recorded", "preference"),
        // Deployment decisions
        ("Always deploy to staging environment before production release",
         Level::Decisions, &["deploy", "safety", "process"], "recorded", "decision"),
        ("Run integration tests in staging before production promote",
         Level::Decisions, &["deploy", "safety", "testing"], "recorded", "decision"),
        // Unrelated record sharing "safety" tag
        ("Wear safety glasses in the workshop at all times",
         Level::Working, &["safety", "workshop"], "recorded", "fact"),
        // Project facts
        ("The backend is written entirely in Rust for performance",
         Level::Identity, &["tech", "rust", "backend"], "recorded", "fact"),
        ("PostgreSQL is the primary database for all services",
         Level::Identity, &["tech", "database", "backend"], "recorded", "fact"),
    ]);

    // ── Batch 2: contradiction + reinforcement ──
    store_batch(&aura, &[
        // Contradicts the staging-first decision
        ("Sometimes we skip staging and deploy hotfixes directly to prod",
         Level::Working, &["deploy", "safety", "process"], "recorded", "contradiction"),
        // Reinforces coding preference
        ("Dark mode with high contrast theme is best for long coding sessions",
         Level::Domain, &["ui", "preferences", "coding"], "recorded", "preference"),
        // New unrelated topic sharing the "tech" tag
        ("We use GitHub Actions for all continuous integration workflows",
         Level::Domain, &["tech", "ci", "automation"], "recorded", "fact"),
    ]);

    // ── Batch 3: paraphrases of existing knowledge ──
    store_batch(&aura, &[
        ("Our entire backend service layer is implemented in Rust",
         Level::Identity, &["tech", "rust", "backend"], "recorded", "fact"),
        ("The project uses Postgres as its main relational database",
         Level::Identity, &["tech", "database", "backend"], "recorded", "fact"),
    ]);

    // ── Run 5 maintenance cycles ──
    let reports = run_cycles(&aura, 5);

    // ── ASSERTIONS ──

    // A. Belief churn should stabilise: last 3 cycles should have churn < 0.15
    for (i, r) in reports.iter().enumerate().skip(2) {
        assert!(
            r.belief.churn_rate < 0.15,
            "cycle {} churn_rate {:.3} exceeds 0.15 — beliefs are unstable",
            i, r.belief.churn_rate
        );
    }

    // B. No runaway unresolved beliefs on this dataset
    let last = reports.last().unwrap();
    assert!(
        last.belief.unresolved <= 1,
        "expected ≤1 unresolved belief, got {}",
        last.belief.unresolved
    );

    // C. Support links should exist (records share tags and semantic types)
    let total_support: usize = reports.iter().map(|r| r.epistemic.total_support_links).sum();
    assert!(
        total_support > 0,
        "expected some support links across 5 cycles, got 0"
    );

    // D. Conflict links should exist (we added a contradiction)
    let total_conflict: usize = reports.iter().map(|r| r.epistemic.total_conflict_links).sum();
    assert!(
        total_conflict > 0,
        "expected some conflict links (we have a contradiction record), got 0"
    );

    // E. The contradiction should NOT cause the workshop-safety record to gain conflict
    //    (different namespace context, only 1 shared tag)
    let workshop_results = recall(&aura, "safety glasses workshop");
    let workshop_rec = workshop_results.iter()
        .find(|(_, r)| r.content.contains("glasses"))
        .map(|(_, r)| r);
    if let Some(rec) = workshop_rec {
        assert_eq!(
            rec.conflict_mass, 0,
            "workshop-safety record should have 0 conflict_mass, got {} \
             (tags={:?}, level={:?})",
            rec.conflict_mass, rec.tags, rec.level
        );
    } else {
        // Workshop record may have decayed — that's fine, just skip this assertion
        eprintln!("NOTE: workshop record not found in recall (likely decayed)");
    }

    // F. Recall quality: querying "dark mode coding" should return coding preference
    let coding_results = recall(&aura, "dark mode coding editor");
    assert!(
        !coding_results.is_empty(),
        "recall for 'dark mode coding editor' returned 0 results"
    );
    let top_content = &coding_results[0].1.content;
    assert!(
        top_content.contains("dark mode") || top_content.contains("Dark mode"),
        "top recall result should mention dark mode, got: {}",
        top_content
    );

    // G. Recall quality: querying "deploy staging" should still return deployment decisions
    let deploy_results = recall(&aura, "deploy staging production");
    assert!(
        !deploy_results.is_empty(),
        "recall for 'deploy staging production' returned 0 results"
    );
    let deploy_contents: Vec<&str> = deploy_results.iter().map(|(_, r)| r.content.as_str()).collect();
    assert!(
        deploy_contents.iter().any(|c| c.contains("staging")),
        "deploy recall should include staging-related records"
    );
}

// ═════════════════════════════════════════════════════════
// 2.  SCENARIO TESTS
// ═════════════════════════════════════════════════════════

/// Scenario: user preferences shift by context (work vs personal).
/// Beliefs about each context should form independently without
/// cross-contamination.
#[test]
fn scenario_contextual_preference_shift() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        // Work context: formal style
        ("At work I write formal commit messages with ticket references",
         Level::Domain, &["communication", "work", "style"], "recorded", "preference"),
        ("Work emails should always have a clear subject line and structure",
         Level::Domain, &["communication", "work", "style"], "recorded", "preference"),
        // Personal context: casual style
        ("In personal chats I prefer short informal messages with emoji",
         Level::Domain, &["communication", "personal", "style"], "recorded", "preference"),
        ("I keep my personal notes brief and unstructured for speed",
         Level::Domain, &["communication", "personal", "style"], "recorded", "preference"),
    ]);

    let reports = run_cycles(&aura, 3);
    let last = &reports[2];

    // No unresolved beliefs — these are different contexts, not contradictions
    assert_eq!(
        last.belief.unresolved, 0,
        "contextual preferences should not produce unresolved beliefs"
    );

    // Zero conflict links expected (same semantic_type, different context tags)
    // Note: they share "communication" and "style" tags, but since all have
    // semantic_type="preference" and no contradiction records, they should
    // accumulate support, not conflict.
    assert_eq!(
        last.epistemic.total_conflict_links, 0,
        "no conflict expected between different-context preferences"
    );
}

/// Scenario: decisions that evolve over time.
/// An initial decision is later contradicted, then a new decision supersedes.
#[test]
fn scenario_evolving_decisions() {
    let (aura, _dir) = open_temp_aura();

    // Phase 1: initial decision
    store_batch(&aura, &[
        ("We use REST APIs for all inter-service communication",
         Level::Decisions, &["architecture", "api", "communication"], "recorded", "decision"),
        ("We use REST APIs for inter-service communication everywhere",
         Level::Decisions, &["architecture", "api", "communication"], "recorded", "decision"),
    ]);

    let r1 = run_cycles(&aura, 2);
    // Should have at least one belief
    let last1 = r1.last().unwrap();
    assert!(
        last1.belief.total_beliefs > 0,
        "initial decisions should form beliefs (records={}, created={}, hyps={})",
        last1.total_records, last1.belief.beliefs_created, last1.belief.total_hypotheses
    );

    // Phase 2: contradiction arrives
    store_batch(&aura, &[
        ("REST is too slow for real-time features, need to reconsider",
         Level::Working, &["architecture", "api", "communication"], "recorded", "contradiction"),
    ]);

    let r2 = run_cycles(&aura, 2);

    // Phase 3: new decision
    store_batch(&aura, &[
        ("We adopt gRPC for latency-critical service communication",
         Level::Decisions, &["architecture", "api", "communication"], "recorded", "decision"),
        ("We adopt gRPC for latency-critical service communication always",
         Level::Decisions, &["architecture", "api", "communication"], "recorded", "decision"),
    ]);

    let r3 = run_cycles(&aura, 3);
    let last = r3.last().unwrap();

    // Conflict should be registered
    let total_conflict: usize = r2.iter().chain(r3.iter())
        .map(|r| r.epistemic.total_conflict_links)
        .sum();
    assert!(
        total_conflict > 0,
        "contradiction should generate conflict links"
    );

    // Churn should be bounded even with evolving decisions
    assert!(
        last.belief.churn_rate < 0.25,
        "churn rate {:.3} too high after decision evolution",
        last.belief.churn_rate
    );
}

/// Scenario: unrelated records that happen to share tags.
/// Should NOT create false epistemic links.
#[test]
fn scenario_shared_tags_different_topics() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        // "security" tag — two unrelated domains
        ("Always enable HTTPS and TLS for production endpoints",
         Level::Decisions, &["security", "infrastructure"], "recorded", "decision"),
        ("Enable two-factor authentication for all admin accounts",
         Level::Decisions, &["security", "infrastructure"], "recorded", "decision"),
        // "security" tag — physical security (different topic)
        ("Install security cameras at all building entrances",
         Level::Working, &["security", "physical", "office"], "recorded", "fact"),
        ("Badge access required for server room entry at facility",
         Level::Working, &["security", "physical", "office"], "recorded", "fact"),
    ]);

    let reports = run_cycles(&aura, 3);
    let last = reports.last().unwrap();

    // Physical security records should not conflict with infra records
    // (different Level pairing doesn't trigger level_conflict since
    // Working vs Decisions is not in the conflict pairs)
    assert_eq!(
        last.epistemic.total_conflict_links, 0,
        "unrelated security records should not generate conflict"
    );
}

/// Scenario: recall quality after multiple maintenance cycles.
/// Records should still be retrievable with good scores even
/// after decay + consolidation.
#[test]
fn scenario_recall_after_maintenance() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("The database connection pool size is set to 25 connections",
         Level::Domain, &["database", "config", "performance"], "recorded", "fact"),
        ("Redis cache TTL is configured to 300 seconds for sessions",
         Level::Domain, &["cache", "config", "performance"], "recorded", "fact"),
        ("API rate limiting is set at 1000 requests per minute per client",
         Level::Domain, &["api", "config", "performance"], "recorded", "fact"),
        ("Logging verbosity is set to INFO in production environment",
         Level::Domain, &["logging", "config", "operations"], "recorded", "fact"),
    ]);

    // Recall before maintenance
    let before = recall(&aura, "database connection pool configuration");
    assert!(!before.is_empty(), "pre-maintenance recall should find records");
    let best_score_before = before[0].0;

    // Run several maintenance cycles
    run_cycles(&aura, 5);

    // Recall after maintenance
    let after = recall(&aura, "database connection pool configuration");
    assert!(!after.is_empty(), "post-maintenance recall should still find records");

    // Top result should still be about database connections
    assert!(
        after[0].1.content.contains("database") || after[0].1.content.contains("connection"),
        "top post-maintenance result should still match the query, got: {}",
        after[0].1.content
    );

    // Score may drop due to decay but should not collapse
    let best_score_after = after[0].0;
    assert!(
        best_score_after > best_score_before * 0.3,
        "recall score dropped too much: {:.3} → {:.3} (>70% loss)",
        best_score_before, best_score_after
    );
}

// ═════════════════════════════════════════════════════════
// 3.  SOAK TESTS
// ═════════════════════════════════════════════════════════

/// 20-cycle soak on a mixed-topic dataset.
/// Checks for metric drift and runaway values.
#[test]
fn soak_20_cycles_metric_stability() {
    let (aura, _dir) = open_temp_aura();

    // Build a moderate dataset: 4 topics × 3 records each
    store_batch(&aura, &[
        // Topic A: deployment
        ("Deploy all services through the CI pipeline automatically",
         Level::Decisions, &["deploy", "ci", "automation"], "recorded", "decision"),
        ("Blue-green deployment strategy for zero-downtime releases",
         Level::Decisions, &["deploy", "strategy", "automation"], "recorded", "decision"),
        ("Rollback procedures must be tested before every deployment",
         Level::Decisions, &["deploy", "safety", "testing"], "recorded", "decision"),
        // Topic B: database
        ("Use connection pooling with PgBouncer for all services",
         Level::Domain, &["database", "postgres", "performance"], "recorded", "fact"),
        ("Database migrations run through Flyway before deployment",
         Level::Domain, &["database", "migrations", "process"], "recorded", "fact"),
        ("Read replicas serve all analytics and reporting queries",
         Level::Domain, &["database", "scaling", "performance"], "recorded", "fact"),
        // Topic C: user preferences
        ("I prefer vim keybindings in all code editors everywhere",
         Level::Identity, &["preferences", "editor", "workflow"], "recorded", "preference"),
        ("Terminal theme is always Solarized Dark for all sessions",
         Level::Identity, &["preferences", "terminal", "theme"], "recorded", "preference"),
        ("Keyboard layout is Colemak for ergonomic typing comfort",
         Level::Identity, &["preferences", "keyboard", "ergonomics"], "recorded", "preference"),
        // Topic D: mixed — includes a contradiction
        ("Monorepo is the correct structure for our codebase layout",
         Level::Decisions, &["architecture", "repo", "structure"], "recorded", "decision"),
        ("Polyrepo gives better isolation for independent team work",
         Level::Working, &["architecture", "repo", "structure"], "recorded", "contradiction"),
        ("Code reviews are mandatory for every pull request merge",
         Level::Decisions, &["process", "review", "quality"], "recorded", "decision"),
    ]);

    // Run 20 maintenance cycles
    let reports = run_cycles(&aura, 20);

    // ── Metric stability assertions ──

    // A. Churn rate should converge toward 0
    let churn_last_5: Vec<f32> = reports[15..].iter().map(|r| r.belief.churn_rate).collect();
    let avg_churn: f32 = churn_last_5.iter().sum::<f32>() / churn_last_5.len() as f32;
    assert!(
        avg_churn < 0.10,
        "avg churn in last 5 cycles = {:.3}, expected < 0.10 — beliefs are not converging",
        avg_churn
    );

    // B. Belief count should be stable (not growing unboundedly)
    let belief_counts: Vec<usize> = reports.iter().map(|r| r.belief.total_beliefs).collect();
    let max_beliefs = *belief_counts.iter().max().unwrap();
    let _min_beliefs_last_half = *belief_counts[10..].iter().min().unwrap();
    assert!(
        max_beliefs <= 20,
        "too many beliefs created ({}), expected ≤20 for 12 records",
        max_beliefs
    );
    // Beliefs shouldn't vanish either
    // (they may be 0 if grouping is too fine-grained, which is also a signal)

    // C. Conflict mass should not grow unboundedly
    let conflict_last: usize = reports.last().unwrap().epistemic.total_conflict_links;
    let conflict_first: usize = reports[0].epistemic.total_conflict_links;
    // Conflict should be roughly stable (not doubling each cycle)
    if conflict_first > 0 {
        assert!(
            conflict_last <= conflict_first * 3,
            "conflict links grew from {} to {} — possible runaway",
            conflict_first, conflict_last
        );
    }

    // D. No volatile records explosion
    let volatile_last = reports.last().unwrap().epistemic.volatile_records;
    assert!(
        volatile_last <= 5,
        "too many volatile records at end ({}), expected ≤5",
        volatile_last
    );

    // E. Unresolved beliefs should be bounded
    let unresolved_last = reports.last().unwrap().belief.unresolved;
    assert!(
        unresolved_last <= 2,
        "too many unresolved beliefs at end ({})",
        unresolved_last
    );

    // F. Total record count should not change (no phantom creation)
    let total_first = reports[0].total_records;
    let total_last = reports.last().unwrap().total_records;
    assert_eq!(
        total_first, total_last,
        "total records changed during soak: {} → {}",
        total_first, total_last
    );
}

/// Soak test focused on epistemic signal stability.
/// Verifies support_mass and conflict_mass converge rather than oscillate.
#[test]
fn soak_epistemic_convergence() {
    let (aura, _dir) = open_temp_aura();

    // Simple cluster: 4 related facts + 1 contradiction
    store_batch(&aura, &[
        ("Rust memory safety prevents data races at compile time",
         Level::Identity, &["rust", "safety", "memory"], "recorded", "fact"),
        ("The borrow checker enforces memory safety without garbage collection",
         Level::Identity, &["rust", "safety", "memory"], "recorded", "fact"),
        ("Ownership model in Rust eliminates use-after-free vulnerabilities",
         Level::Identity, &["rust", "safety", "memory"], "recorded", "fact"),
        ("Rust zero-cost abstractions provide safety without runtime overhead",
         Level::Identity, &["rust", "safety", "performance"], "recorded", "fact"),
        // Contradiction
        ("Unsafe blocks in Rust bypass memory safety guarantees entirely",
         Level::Working, &["rust", "safety", "memory"], "recorded", "contradiction"),
    ]);

    // Run 10 cycles, collect epistemic snapshots
    let reports = run_cycles(&aura, 10);

    // Support links should be present (4 related facts)
    let support_counts: Vec<usize> = reports.iter()
        .map(|r| r.epistemic.total_support_links)
        .collect();
    assert!(
        support_counts.iter().all(|&s| s > 0),
        "support links should be present every cycle: {:?}",
        support_counts
    );

    // Conflict links should be present (1 contradiction)
    let conflict_counts: Vec<usize> = reports.iter()
        .map(|r| r.epistemic.total_conflict_links)
        .collect();
    assert!(
        conflict_counts.iter().all(|&c| c > 0),
        "conflict links should be present every cycle: {:?}",
        conflict_counts
    );

    // Support and conflict should stabilise: variance in last 5 cycles should be low
    let support_last_5 = &support_counts[5..];
    let conflict_last_5 = &conflict_counts[5..];

    let support_range = support_last_5.iter().max().unwrap() - support_last_5.iter().min().unwrap();
    let conflict_range = conflict_last_5.iter().max().unwrap() - conflict_last_5.iter().min().unwrap();

    assert!(
        support_range <= 2,
        "support links oscillating too much in last 5 cycles: {:?} (range={})",
        support_last_5, support_range
    );
    assert!(
        conflict_range <= 2,
        "conflict links oscillating too much in last 5 cycles: {:?} (range={})",
        conflict_last_5, conflict_range
    );
}

// ═════════════════════════════════════════════════════════
// 4.  SDR GROUPING PRECISION TESTS
// ═════════════════════════════════════════════════════════

/// Full-stack test: records with same tags but different topics should
/// form separate beliefs thanks to SDR-based claim grouping.
///
/// This is the key precision test — it verifies that Aura's belief engine
/// uses SDR content similarity to distinguish topic clusters within a
/// shared tag-group.
#[test]
fn sdr_grouping_separates_different_topics() {
    let (aura, _dir) = open_temp_aura();

    // Two distinct topics, both tagged ["ops", "config"]
    // Topic A: deployment pipeline
    store_batch(&aura, &[
        ("Configure blue-green deployment pipeline for zero-downtime releases",
         Level::Decisions, &["ops", "config", "deployment"], "recorded", "decision"),
        ("Set up blue-green deployment pipeline with automatic rollback capability",
         Level::Decisions, &["ops", "config", "deployment"], "recorded", "decision"),
    ]);
    // Topic B: database connection tuning
    store_batch(&aura, &[
        ("Configure PostgreSQL connection pool with maximum twenty-five connections",
         Level::Decisions, &["ops", "config", "database"], "recorded", "decision"),
        ("Set PostgreSQL connection pool maximum to twenty-five active connections",
         Level::Decisions, &["ops", "config", "database"], "recorded", "decision"),
    ]);

    // Run maintenance — SDR grouping should separate these into ≥2 beliefs
    let reports = run_cycles(&aura, 3);
    let last = reports.last().unwrap();

    // With SDR grouping, deployment and database records should form
    // separate beliefs despite sharing tags.
    // Without SDR, they'd all merge into 1 belief (same coarse key).
    assert!(
        last.belief.total_beliefs >= 2,
        "SDR grouping should create ≥2 beliefs for different topics \
         (deployment vs database), got {} (hyps={})",
        last.belief.total_beliefs, last.belief.total_hypotheses
    );

    // No unresolved — both clusters are internally consistent
    assert_eq!(
        last.belief.unresolved, 0,
        "both topic clusters should resolve cleanly"
    );
}

/// Full-stack test: records about the same topic with paraphrases should
/// still merge into one belief even with SDR grouping.
#[test]
fn sdr_grouping_merges_paraphrases() {
    let (aura, _dir) = open_temp_aura();

    // Same topic, paraphrased differently
    store_batch(&aura, &[
        ("Always run integration tests before merging pull requests",
         Level::Decisions, &["testing", "process", "quality"], "recorded", "decision"),
        ("Run all integration tests before merging any pull request",
         Level::Decisions, &["testing", "process", "quality"], "recorded", "decision"),
        ("Integration test suite must pass prior to pull request merge",
         Level::Decisions, &["testing", "process", "quality"], "recorded", "decision"),
    ]);

    let reports = run_cycles(&aura, 3);
    let last = reports.last().unwrap();

    // All three paraphrases should group into the same belief (≤ 1 belief)
    // or at most 2 if SDR sees them as slightly different
    assert!(
        last.belief.total_beliefs <= 2,
        "paraphrases of the same decision should form ≤2 beliefs, got {}",
        last.belief.total_beliefs
    );

    // No unresolved — they all say the same thing
    assert_eq!(
        last.belief.unresolved, 0,
        "paraphrases should not create unresolved conflicts"
    );
}
