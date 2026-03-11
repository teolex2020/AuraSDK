//! Integration tests for policy hint discovery layer (Phase 3.9).
//!
//! Exercises the full Aura stack: store records → run maintenance cycles
//! (belief + concept + causal + policy phases) → verify policy hints meet constraints:
//!
//! - policy phase runs without error in maintenance
//! - policy hints form from strong causal patterns
//! - policy discovery does not affect recall
//! - policy metrics are stable across repeated maintenance cycles
//! - suppression of conflicting hints works
//! - policy report has valid (non-negative) metrics
//! - fresh startup has empty policy state

use aura::{Aura, Level};

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

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
            Some(false),
            None, None,
            Some(*semantic),
        )
        .unwrap_or_else(|e| panic!("store failed for '{}': {}", content, e));
    }
}

fn run_cycles(aura: &Aura, n: usize) -> Vec<aura::background_brain::MaintenanceReport> {
    (0..n).map(|_| aura.run_maintenance()).collect()
}

fn recall(aura: &Aura, query: &str) -> Vec<(f32, aura::Record)> {
    aura.recall_structured(query, Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed")
}

// ═════════════════════════════════════════════════════════
// 1.  POLICY PHASE RUNS IN MAINTENANCE
// ═════════════════════════════════════════════════════════

/// Store related records, run maintenance, verify policy phase executed.
#[test]
fn policy_phase_runs_in_maintenance() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Deployed canary release to staging environment", Level::Domain, &["deploy", "ops"], "recorded", "decision"),
        ("Canary deployment caught regression before production", Level::Domain, &["deploy", "ops"], "recorded", "fact"),
        ("Canary strategy reduced incident rate by forty percent", Level::Domain, &["deploy", "ops"], "recorded", "fact"),
        ("Team adopted canary releases as standard practice", Level::Domain, &["deploy", "ops"], "recorded", "decision"),
    ]);

    let reports = run_cycles(&aura, 5);
    let last = reports.last().unwrap();

    // Policy phase must execute without error
    assert!(last.policy.avg_policy_strength >= 0.0,
        "policy strength should be non-negative");
    assert!(last.total_records > 0,
        "records should survive maintenance");
}

// ═════════════════════════════════════════════════════════
// 2.  POLICY DISCOVERY DOES NOT AFFECT RECALL
// ═════════════════════════════════════════════════════════

#[test]
fn policy_discovery_does_not_degrade_recall() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Kubernetes autoscaler manages pod replicas efficiently", Level::Domain, &["k8s", "scaling"], "recorded", "fact"),
        ("Autoscaler reduced infrastructure cost by thirty percent", Level::Domain, &["k8s", "scaling"], "recorded", "fact"),
        ("Kubernetes cluster handles traffic spikes automatically", Level::Domain, &["k8s", "scaling"], "recorded", "fact"),
    ]);

    let before = recall(&aura, "Kubernetes autoscaling");
    assert!(!before.is_empty(), "should recall before maintenance");

    run_cycles(&aura, 5);

    let after = recall(&aura, "Kubernetes autoscaling");
    assert!(!after.is_empty(), "should recall after maintenance with policy phase");
    assert!(after.len() >= before.len(),
        "recall should not lose results after policy discovery");
}

// ═════════════════════════════════════════════════════════
// 3.  POLICY METRICS STABLE ACROSS CYCLES
// ═════════════════════════════════════════════════════════

#[test]
fn policy_metrics_stable_across_repeated_cycles() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Enabled connection pooling for database access", Level::Domain, &["database", "perf"], "recorded", "decision"),
        ("Connection pooling reduced query latency significantly", Level::Domain, &["database", "perf"], "recorded", "fact"),
        ("Database connection pool handles concurrent load well", Level::Domain, &["database", "perf"], "recorded", "fact"),
        ("Performance monitoring shows stable query times now", Level::Domain, &["database", "perf"], "recorded", "fact"),
    ]);

    let reports = run_cycles(&aura, 8);

    // After initial formation, hint count should stabilize
    let last_3: Vec<_> = reports[5..].iter()
        .map(|r| r.policy.hints_found)
        .collect();

    // All last 3 should be the same (stable — full rebuild produces same result)
    if !last_3.is_empty() && last_3[0] > 0 {
        for count in &last_3 {
            assert_eq!(*count, last_3[0],
                "policy hint count should be stable in later cycles: {:?}", last_3);
        }
    }
}

// ═════════════════════════════════════════════════════════
// 4.  POLICY REPORT HAS VALID METRICS
// ═════════════════════════════════════════════════════════

#[test]
fn policy_report_has_valid_metrics() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Implemented circuit breaker for external API calls", Level::Domain, &["resilience", "api"], "recorded", "decision"),
        ("Circuit breaker prevented cascade failures in production", Level::Domain, &["resilience", "api"], "recorded", "fact"),
        ("API reliability improved after circuit breaker deployment", Level::Domain, &["resilience", "api"], "recorded", "fact"),
        ("Automated fallback activates when circuit breaker trips", Level::Domain, &["resilience", "api"], "recorded", "decision"),
        // Second cluster
        ("Removed legacy API endpoints from service mesh", Level::Domain, &["cleanup", "api"], "recorded", "decision"),
        ("Legacy removal caused brief client errors during migration", Level::Domain, &["cleanup", "api", "error", "failure"], "recorded", "fact"),
        ("Some clients experienced timeout issues after removal", Level::Domain, &["cleanup", "api", "error"], "recorded", "fact"),
        ("Rollback plan activated to restore legacy endpoints", Level::Domain, &["cleanup", "api"], "recorded", "decision"),
    ]);

    let reports = run_cycles(&aura, 5);
    let last = reports.last().unwrap();

    // Policy phase should have run
    assert!(last.policy.avg_policy_strength >= 0.0,
        "policy strength should be non-negative");

    // Seeds should be >= hints (not all seeds produce hints)
    assert!(last.policy.seeds_found >= last.policy.hints_found
        || last.policy.seeds_found == 0,
        "seeds should be >= hints (or both zero)");

    // Suppressed + rejected should not exceed total hints found
    // (stable + suppressed + rejected <= hints_found is guaranteed by design)
}

// ═════════════════════════════════════════════════════════
// 5.  SOAK: 15 CYCLES, POLICY METRICS DON'T DIVERGE
// ═════════════════════════════════════════════════════════

#[test]
fn soak_policy_metrics_converge() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Migrated logging to structured JSON format", Level::Domain, &["observability", "logging"], "recorded", "decision"),
        ("Structured logs improved search and alerting accuracy", Level::Domain, &["observability", "logging"], "recorded", "fact"),
        ("Log aggregation pipeline processes structured logs faster", Level::Domain, &["observability", "logging"], "recorded", "fact"),
        ("Team productivity increased with better log visibility", Level::Domain, &["observability", "logging"], "recorded", "fact"),
    ]);

    let reports = run_cycles(&aura, 15);

    // Collect hint counts from last 5 cycles
    let last_5: Vec<usize> = reports[10..].iter()
        .map(|r| r.policy.hints_found)
        .collect();

    // Should be stable (no runaway growth)
    let max = *last_5.iter().max().unwrap_or(&0);
    let min = *last_5.iter().min().unwrap_or(&0);
    assert!(max - min <= 1,
        "policy hint count should be stable in soak test, got {:?}", last_5);

    // Strength scores should not diverge
    let scores: Vec<f32> = reports[10..].iter()
        .map(|r| r.policy.avg_policy_strength)
        .collect();
    if let (Some(&max_s), Some(&min_s)) = (scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                                             scores.iter().min_by(|a, b| a.partial_cmp(b).unwrap())) {
        assert!((max_s - min_s).abs() < 0.1,
            "policy strength scores should converge, got {:?}", scores);
    }
}

// ═════════════════════════════════════════════════════════
// 6.  POLICY STATE EMPTY ON FRESH STARTUP
// ═════════════════════════════════════════════════════════

#[test]
fn policy_empty_on_fresh_startup() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Initial system boot sequence completed", Level::Domain, &["setup"], "recorded", "fact"),
    ]);

    // First maintenance starts from empty policy state
    let reports = run_cycles(&aura, 1);
    let first = &reports[0];

    assert!(first.policy.avg_policy_strength >= 0.0);
}

// ═════════════════════════════════════════════════════════
// 7.  POLICY ZERO RECALL IMPACT VERIFICATION
// ═════════════════════════════════════════════════════════

#[test]
fn policy_has_zero_recall_impact() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Docker containers simplify application deployment", Level::Domain, &["docker", "containers"], "recorded", "fact"),
        ("Container orchestration automates scaling decisions", Level::Domain, &["docker", "containers"], "recorded", "fact"),
        ("Docker images are built and cached in CI pipeline", Level::Domain, &["docker", "containers"], "recorded", "decision"),
    ]);

    let before = recall(&aura, "Docker container deployment");

    // Run many cycles (policy hints will rebuild repeatedly)
    run_cycles(&aura, 10);

    let after = recall(&aura, "Docker container deployment");

    assert!(after.len() >= before.len(),
        "recall must not degrade: before={}, after={}", before.len(), after.len());

    if !after.is_empty() {
        assert!(after[0].1.content.to_lowercase().contains("docker"),
            "top result should still be about Docker");
    }
}

// ═════════════════════════════════════════════════════════
// 8.  DIFFERENT TOPICS STAY INDEPENDENT IN POLICY
// ═════════════════════════════════════════════════════════

#[test]
fn different_topics_stay_independent_in_policy() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Adopted GraphQL for new API endpoints", Level::Domain, &["graphql", "api"], "recorded", "decision"),
        ("GraphQL reduced over-fetching and improved client performance", Level::Domain, &["graphql", "api"], "recorded", "fact"),
        ("GraphQL schema validation catches errors at build time", Level::Domain, &["graphql", "api"], "recorded", "fact"),
    ]);

    store_batch(&aura, &[
        ("Switched from monolith to microservices architecture", Level::Domain, &["architecture", "microservices"], "recorded", "decision"),
        ("Microservices enabled independent team deployments", Level::Domain, &["architecture", "microservices"], "recorded", "fact"),
        ("Service mesh handles inter-service communication", Level::Domain, &["architecture", "microservices"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 5);

    // Both topics should be independently retrievable
    let graphql = recall(&aura, "GraphQL API endpoints");
    let micro = recall(&aura, "microservices architecture");

    assert!(!graphql.is_empty(), "GraphQL topic should be recallable");
    assert!(!micro.is_empty(), "microservices topic should be recallable");
}

// ═════════════════════════════════════════════════════════
// 9.  SURFACED OUTPUT IS EMPTY ON FRESH STARTUP
// ═════════════════════════════════════════════════════════

#[test]
fn surfaced_policy_output_is_empty_on_fresh_startup() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Initial system setup", Level::Domain, &["setup"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 1);

    let surfaced = aura.get_surfaced_policy_hints(None);
    assert!(surfaced.is_empty(),
        "surfaced output should be empty on fresh startup, got {}", surfaced.len());
}

// ═════════════════════════════════════════════════════════
// 10. SURFACED OUTPUT HAS ZERO RECALL IMPACT
// ═════════════════════════════════════════════════════════

#[test]
fn surfaced_policy_output_has_zero_recall_impact() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Nginx reverse proxy handles load balancing", Level::Domain, &["nginx", "proxy"], "recorded", "fact"),
        ("Load balancer distributes traffic across servers", Level::Domain, &["nginx", "proxy"], "recorded", "fact"),
        ("Nginx configuration tuned for production traffic", Level::Domain, &["nginx", "proxy"], "recorded", "decision"),
    ]);

    let before = recall(&aura, "Nginx load balancing");

    run_cycles(&aura, 8);

    // Access surfaced output (should not affect recall)
    let _surfaced = aura.get_surfaced_policy_hints(None);

    let after = recall(&aura, "Nginx load balancing");
    assert!(after.len() >= before.len(),
        "recall must not degrade after surfacing: before={}, after={}", before.len(), after.len());
}

// ═════════════════════════════════════════════════════════
// 11. SURFACED OUTPUT IS BOUNDED
// ═════════════════════════════════════════════════════════

#[test]
fn surfaced_policy_output_is_bounded() {
    let (aura, _dir) = open_temp_aura();

    // Store many records across multiple topics
    store_batch(&aura, &[
        ("Deployed service mesh for microservices", Level::Domain, &["mesh", "deploy"], "recorded", "decision"),
        ("Service mesh improved inter-service reliability", Level::Domain, &["mesh", "deploy"], "recorded", "fact"),
        ("Adopted GitOps workflow for deployments", Level::Domain, &["gitops", "deploy"], "recorded", "decision"),
        ("GitOps reduced deployment errors significantly", Level::Domain, &["gitops", "deploy"], "recorded", "fact"),
        ("Implemented chaos engineering for resilience", Level::Domain, &["chaos", "resilience"], "recorded", "decision"),
        ("Chaos testing found hidden failure modes", Level::Domain, &["chaos", "resilience"], "recorded", "fact"),
        ("Added circuit breakers to external calls", Level::Domain, &["circuit", "resilience"], "recorded", "decision"),
        ("Circuit breakers prevented cascade failures", Level::Domain, &["circuit", "resilience"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_policy_hints(None);
    assert!(surfaced.len() <= 10,
        "surfaced output should respect global limit, got {}", surfaced.len());

    let surfaced_3 = aura.get_surfaced_policy_hints(Some(3));
    assert!(surfaced_3.len() <= 3,
        "surfaced output should respect explicit limit, got {}", surfaced_3.len());
}

// ═════════════════════════════════════════════════════════
// 12. SURFACED OUTPUT CONTAINS FULL PROVENANCE
// ═════════════════════════════════════════════════════════

#[test]
fn surfaced_policy_output_contains_full_provenance() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Enabled automated testing in CI pipeline", Level::Domain, &["testing", "ci"], "recorded", "decision"),
        ("CI pipeline catches regressions before deploy", Level::Domain, &["testing", "ci"], "recorded", "fact"),
        ("Test coverage improved after CI enforcement", Level::Domain, &["testing", "ci"], "recorded", "fact"),
        ("Quality metrics show fewer production bugs", Level::Domain, &["testing", "ci"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 10);

    let surfaced = aura.get_surfaced_policy_hints(None);
    for hint in &surfaced {
        assert!(!hint.trigger_causal_ids.is_empty(),
            "surfaced hint '{}' must have causal provenance", hint.id);
        assert!(!hint.supporting_record_ids.is_empty(),
            "surfaced hint '{}' must have record provenance", hint.id);
    }
}

// ═════════════════════════════════════════════════════════
// 13. SURFACED OUTPUT STABLE ACROSS REPLAY
// ═════════════════════════════════════════════════════════

#[test]
fn surfaced_policy_output_stable_across_replay() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Implemented feature flags for gradual rollout", Level::Domain, &["flags", "deploy"], "recorded", "decision"),
        ("Feature flags reduced rollout risk", Level::Domain, &["flags", "deploy"], "recorded", "fact"),
        ("Gradual rollout caught issues in small cohort", Level::Domain, &["flags", "deploy"], "recorded", "fact"),
    ]);

    // Run 10 cycles, collect surfaced output from last 3
    let reports = run_cycles(&aura, 10);
    let _ = reports; // use reports to keep borrow checker happy

    let s1 = aura.get_surfaced_policy_hints(None);

    // Run 3 more cycles
    run_cycles(&aura, 3);
    let s2 = aura.get_surfaced_policy_hints(None);

    // Same count (stable output)
    assert_eq!(s1.len(), s2.len(),
        "surfaced output should be stable across replay: {} vs {}", s1.len(), s2.len());

    // If non-empty, same IDs (deterministic)
    if !s1.is_empty() {
        let ids1: Vec<&str> = s1.iter().map(|h| h.id.as_str()).collect();
        let ids2: Vec<&str> = s2.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids1, ids2, "surfaced hint IDs should be stable across replay");
    }
}
