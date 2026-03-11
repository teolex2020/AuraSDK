//! Step 9 — Cross-layer evaluation with belief rerank enabled (Limited mode).
//!
//! Re-runs the core cross-layer scenarios from cross_layer_eval.rs with
//! `BeliefRerankMode::Limited` active. Validates that:
//!
//!   1. All cross-layer invariants still hold under active reranking
//!   2. Recall results remain stable (overlap with baseline)
//!   3. Rerank reports show bounded behavior (caps respected)
//!   4. No contradiction leakage or provenance gaps introduced
//!   5. Churn bounds not violated by reranking activity
//!
//! This is NOT a repeat of cross_layer_eval.rs — it specifically tests the
//! interaction between the promoted belief reranking and the full stack.

use aura::{Aura, Level};
use aura::background_brain::MaintenanceReport;
use aura::recall::{BeliefRerankMode, LimitedRerankReport};

// ═══════════════════════════════════════════════════════════
// Harness infrastructure (mirrors cross_layer_eval.rs)
// ═══════════════════════════════════════════════════════════

struct ScenarioRecord {
    content: &'static str,
    level: Level,
    tags: &'static [&'static str],
    source_type: &'static str,
    semantic_type: &'static str,
}

#[derive(Debug)]
#[allow(dead_code)]
struct CycleSnapshot {
    cycle: usize,
    total_records: usize,
    belief_count: usize,
    resolved_beliefs: usize,
    unresolved_beliefs: usize,
    belief_churn: f32,
    concept_count: usize,
    stable_concepts: usize,
    concept_churn: f32,
    causal_count: usize,
    stable_causal: usize,
    causal_churn: f32,
    policy_count: usize,
    stable_policy: usize,
    suppressed_policy: usize,
    policy_churn: f32,
    total_ms: f64,
    causal_missing_provenance: usize,
    policy_missing_provenance: usize,
}

/// Per-query rerank comparison between Off and Limited modes.
#[derive(Debug)]
#[allow(dead_code)]
struct RerankComparison {
    query: String,
    baseline_count: usize,
    limited_count: usize,
    /// Fraction of record IDs shared between baseline top-k and limited top-k.
    top_k_overlap: f32,
    /// From the LimitedRerankReport.
    report: LimitedRerankReport,
}

// ── Helpers ──

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

fn store_scenario(aura: &Aura, records: &[ScenarioRecord]) {
    for rec in records {
        aura.store(
            rec.content,
            Some(rec.level),
            Some(rec.tags.iter().map(|t| t.to_string()).collect()),
            None, None,
            Some(rec.source_type),
            None,
            Some(false),
            None, None,
            Some(rec.semantic_type),
        )
        .unwrap_or_else(|e| panic!("store failed for '{}': {}", rec.content, e));
    }
}

fn recall_off(aura: &Aura, query: &str, top_k: usize) -> Vec<(f32, aura::Record)> {
    // Ensure Off mode for baseline
    aura.set_belief_rerank_mode(BeliefRerankMode::Off);
    aura.recall_structured(query, Some(top_k), Some(0.0), Some(true), None, None)
        .expect("recall failed")
}

fn recall_limited_with_report(
    aura: &Aura,
    query: &str,
    top_k: usize,
) -> (Vec<(f32, aura::Record)>, LimitedRerankReport) {
    aura.recall_structured_with_rerank_report(
        query, Some(top_k), Some(0.0), Some(true), None, None,
    )
    .expect("rerank report recall failed")
}

fn compute_overlap(
    baseline: &[(f32, aura::Record)],
    limited: &[(f32, aura::Record)],
) -> f32 {
    if baseline.is_empty() && limited.is_empty() {
        return 1.0;
    }
    if baseline.is_empty() || limited.is_empty() {
        return 0.0;
    }
    let baseline_ids: std::collections::HashSet<&str> = baseline.iter()
        .map(|(_, r)| r.id.as_str())
        .collect();
    let limited_ids: std::collections::HashSet<&str> = limited.iter()
        .map(|(_, r)| r.id.as_str())
        .collect();
    let shared = baseline_ids.intersection(&limited_ids).count();
    shared as f32 / baseline_ids.len().max(limited_ids.len()) as f32
}

fn compare_rerank(
    aura: &Aura,
    query: &str,
    top_k: usize,
) -> RerankComparison {
    let baseline = recall_off(aura, query, top_k);
    let (limited, report) = recall_limited_with_report(aura, query, top_k);
    let overlap = compute_overlap(&baseline, &limited);

    RerankComparison {
        query: query.to_string(),
        baseline_count: baseline.len(),
        limited_count: limited.len(),
        top_k_overlap: overlap,
        report,
    }
}

fn take_snapshot(aura: &Aura, cycle: usize, report: &MaintenanceReport) -> CycleSnapshot {
    let beliefs = aura.get_beliefs(None);
    let concepts = aura.get_concepts(None);
    let causal = aura.get_causal_patterns(None);
    let policy = aura.get_policy_hints(None);

    let resolved_beliefs = beliefs.iter()
        .filter(|b| b.state == aura::belief::BeliefState::Resolved)
        .count();
    let unresolved_beliefs = beliefs.iter()
        .filter(|b| b.state == aura::belief::BeliefState::Unresolved)
        .count();
    let stable_concepts = concepts.iter()
        .filter(|c| c.state == aura::concept::ConceptState::Stable)
        .count();
    let stable_causal = causal.iter()
        .filter(|p| p.state == aura::causal::CausalState::Stable)
        .count();
    let stable_policy = policy.iter()
        .filter(|h| h.state == aura::policy::PolicyState::Stable)
        .count();
    let suppressed_policy = policy.iter()
        .filter(|h| h.state == aura::policy::PolicyState::Suppressed)
        .count();
    let causal_missing_provenance = causal.iter()
        .filter(|p| p.state == aura::causal::CausalState::Stable)
        .filter(|p| p.cause_record_ids.is_empty() || p.effect_record_ids.is_empty())
        .count();
    let policy_missing_provenance = policy.iter()
        .filter(|h| h.state == aura::policy::PolicyState::Stable)
        .filter(|h| h.trigger_causal_ids.is_empty() || h.supporting_record_ids.is_empty())
        .count();

    CycleSnapshot {
        cycle,
        total_records: report.total_records,
        belief_count: beliefs.len(),
        resolved_beliefs,
        unresolved_beliefs,
        belief_churn: report.stability.belief_churn,
        concept_count: concepts.len(),
        stable_concepts,
        concept_churn: report.stability.concept_churn,
        causal_count: causal.len(),
        stable_causal,
        causal_churn: report.stability.causal_churn,
        policy_count: policy.len(),
        stable_policy,
        suppressed_policy,
        policy_churn: report.stability.policy_churn,
        total_ms: report.timings.total_ms,
        causal_missing_provenance,
        policy_missing_provenance,
    }
}

// ── Invariant assertions ──

fn assert_churn_bounded(snapshots: &[CycleSnapshot], warmup: usize,
                         belief_max: f32, concept_max: f32,
                         causal_max: f32, policy_max: f32) {
    for snap in snapshots.iter().skip(warmup) {
        assert!(snap.belief_churn <= belief_max,
            "cycle {}: belief_churn {} > max {}", snap.cycle, snap.belief_churn, belief_max);
        assert!(snap.concept_churn <= concept_max,
            "cycle {}: concept_churn {} > max {}", snap.cycle, snap.concept_churn, concept_max);
        assert!(snap.causal_churn <= causal_max,
            "cycle {}: causal_churn {} > max {}", snap.cycle, snap.causal_churn, causal_max);
        assert!(snap.policy_churn <= policy_max,
            "cycle {}: policy_churn {} > max {}", snap.cycle, snap.policy_churn, policy_max);
    }
}

fn assert_no_provenance_gaps(snapshots: &[CycleSnapshot]) {
    for snap in snapshots {
        assert_eq!(snap.causal_missing_provenance, 0,
            "cycle {}: {} stable causal missing provenance", snap.cycle, snap.causal_missing_provenance);
        assert_eq!(snap.policy_missing_provenance, 0,
            "cycle {}: {} stable policy missing provenance", snap.cycle, snap.policy_missing_provenance);
    }
}

fn assert_timings_valid(reports: &[MaintenanceReport]) {
    for (i, r) in reports.iter().enumerate() {
        assert!(r.timings.total_ms >= 0.0, "cycle {}: negative total_ms {}", i, r.timings.total_ms);
        assert!(r.timings.belief_ms >= 0.0, "cycle {}: negative belief_ms {}", i, r.timings.belief_ms);
    }
}

fn assert_suppression_bounded(snapshots: &[CycleSnapshot]) {
    for snap in snapshots {
        assert!(snap.suppressed_policy <= snap.policy_count,
            "cycle {}: suppressed {} > total {}", snap.cycle, snap.suppressed_policy, snap.policy_count);
    }
}

/// Assert all rerank comparisons respect caps and have decent overlap.
fn assert_rerank_bounded(comparisons: &[RerankComparison]) {
    for cmp in comparisons {
        // Positional shift caps
        assert!(cmp.report.max_up_shift <= 2,
            "query '{}': max_up_shift {} > 2", cmp.query, cmp.report.max_up_shift);
        assert!(cmp.report.max_down_shift <= 2,
            "query '{}': max_down_shift {} > 2", cmp.query, cmp.report.max_down_shift);

        // If reranking was applied, overlap should be high (≥ 0.70)
        if cmp.report.was_applied {
            assert!(cmp.top_k_overlap >= 0.70,
                "query '{}': overlap {} < 0.70 with rerank applied", cmp.query, cmp.top_k_overlap);
        }

        // Result count should not change (rerank only reorders, never drops)
        assert_eq!(cmp.baseline_count, cmp.limited_count,
            "query '{}': baseline {} != limited {} result count",
            cmp.query, cmp.baseline_count, cmp.limited_count);
    }
}

// ═══════════════════════════════════════════════════════════
// Scenario data (same as cross_layer_eval.rs)
// ═══════════════════════════════════════════════════════════

fn scenario_a_records() -> Vec<ScenarioRecord> {
    vec![
        ScenarioRecord { content: "I prefer dark mode in my IDE for coding", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark mode reduces eye strain during long coding sessions", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "My preferred editor theme is dark with high contrast", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark backgrounds work better for evening programming", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Rust compiler gives helpful error messages", level: Level::Domain, tags: &["rust", "tooling"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Cargo build system handles dependencies well", level: Level::Domain, tags: &["rust", "tooling"], source_type: "recorded", semantic_type: "fact" },
    ]
}

fn scenario_b_records() -> Vec<ScenarioRecord> {
    vec![
        ScenarioRecord { content: "Deployed canary release to staging before production", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Canary testing caught no regressions in staging", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Production deploy succeeded after canary approval", level: Level::Domain, tags: &["deploy", "production"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Canary release process improved deployment safety", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Pushed hotfix directly to production without staging", level: Level::Domain, tags: &["deploy", "production", "hotfix"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Hotfix caused unexpected error rate spike in production", level: Level::Domain, tags: &["deploy", "production", "incident"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Emergency rollback was required after hotfix failure", level: Level::Domain, tags: &["deploy", "production", "incident"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Post-incident review recommended mandatory staging", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Staging environments reduce production incident rate", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Deploy pipeline includes automated smoke tests", level: Level::Domain, tags: &["deploy", "testing"], source_type: "recorded", semantic_type: "fact" },
    ]
}

fn scenario_c_records() -> Vec<ScenarioRecord> {
    vec![
        ScenarioRecord { content: "I use dark mode in my code editor for comfort", level: Level::Domain, tags: &["editor", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark editor themes help focus during programming", level: Level::Domain, tags: &["editor", "theme"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "VS Code dark theme is my primary editor setting", level: Level::Domain, tags: &["editor", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "I prefer light mode when reading documentation", level: Level::Domain, tags: &["docs", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Light backgrounds improve readability for long documents", level: Level::Domain, tags: &["docs", "theme"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Documentation sites look better with light themes", level: Level::Domain, tags: &["docs", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
    ]
}

fn scenario_d_records() -> Vec<ScenarioRecord> {
    vec![
        ScenarioRecord { content: "Blue-green deployment minimizes downtime", level: Level::Domain, tags: &["deploy", "infrastructure"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Automated rollback triggers on error rate threshold", level: Level::Domain, tags: &["deploy", "infrastructure"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Deploy pipeline runs integration tests before release", level: Level::Domain, tags: &["deploy", "testing"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Sans-serif fonts improve UI readability on screens", level: Level::Domain, tags: &["ui", "design"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Consistent spacing in UI components reduces cognitive load", level: Level::Domain, tags: &["ui", "design"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Accessible color contrast ratios are mandatory", level: Level::Domain, tags: &["ui", "accessibility"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Structured JSON logs enable efficient aggregation", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Log levels should follow severity conventions", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Centralized logging reduces debugging time", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "All API endpoints require authentication tokens", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Rate limiting prevents abuse of public endpoints", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Security headers configured on all HTTP responses", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "decision" },
    ]
}

// ═══════════════════════════════════════════════════════════
// Test A: Stable Preference — rerank should not disturb
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_scenario_a_stable_preference_with_rerank() {
    let queries = &["dark mode editor preference", "Rust compiler tooling"];
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_a_records());

    // Run maintenance to build belief layer
    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for i in 0..8 {
        let report = aura.run_maintenance();
        snapshots.push(take_snapshot(&aura, i, &report));
        reports.push(report);
    }

    // Enable Limited mode
    aura.set_belief_rerank_mode(BeliefRerankMode::Limited);
    assert_eq!(aura.get_belief_rerank_mode(), BeliefRerankMode::Limited);

    // Cross-layer invariants still hold
    assert_timings_valid(&reports);
    assert_no_provenance_gaps(&snapshots);
    assert_suppression_bounded(&snapshots);
    assert_churn_bounded(&snapshots, 2, 0.15, 0.20, 0.20, 0.20);

    // Rerank comparison: Off vs Limited
    let comparisons: Vec<RerankComparison> = queries.iter()
        .map(|q| compare_rerank(&aura, q, 20))
        .collect();
    assert_rerank_bounded(&comparisons);

    // Stable preference data: low belief coverage expected, rerank should be mild
    for cmp in &comparisons {
        // Overlap must be very high for stable preference data
        assert!(cmp.top_k_overlap >= 0.80,
            "query '{}': stable preference overlap {} < 0.80", cmp.query, cmp.top_k_overlap);
    }

    let last = snapshots.last().unwrap();
    assert!(last.total_records >= 1, "records should survive");
    assert!(last.policy_count <= 5, "policy bounded for preferences: got {}", last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Test B: Deploy/Safety Chain — rerank under causal data
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_scenario_b_deploy_chain_with_rerank() {
    let queries = &["deploy staging canary", "hotfix production incident"];
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_b_records());

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for i in 0..8 {
        let report = aura.run_maintenance();
        snapshots.push(take_snapshot(&aura, i, &report));
        reports.push(report);
    }

    aura.set_belief_rerank_mode(BeliefRerankMode::Limited);

    // Cross-layer invariants
    assert_timings_valid(&reports);
    assert_no_provenance_gaps(&snapshots);
    assert_suppression_bounded(&snapshots);
    assert_churn_bounded(&snapshots, 3, 0.15, 0.25, 0.25, 0.25);

    // Rerank comparison
    let comparisons: Vec<RerankComparison> = queries.iter()
        .map(|q| compare_rerank(&aura, q, 20))
        .collect();
    assert_rerank_bounded(&comparisons);

    let last = snapshots.last().unwrap();
    assert!(last.total_records >= 1, "records should survive");
    assert!(last.policy_count <= 10, "policy bounded: got {}", last.policy_count);
    if last.policy_count > 0 {
        assert!(last.suppressed_policy <= last.policy_count / 2 + 1,
            "suppression not dominating: {} suppressed of {}", last.suppressed_policy, last.policy_count);
    }
}

// ═══════════════════════════════════════════════════════════
// Test C: Contextual Preference — no false conflict with rerank
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_scenario_c_contextual_with_rerank() {
    let queries = &["dark mode code editor", "light mode documentation reading"];
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_c_records());

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for i in 0..8 {
        let report = aura.run_maintenance();
        snapshots.push(take_snapshot(&aura, i, &report));
        reports.push(report);
    }

    aura.set_belief_rerank_mode(BeliefRerankMode::Limited);

    assert_timings_valid(&reports);
    assert_no_provenance_gaps(&snapshots);
    assert_suppression_bounded(&snapshots);
    assert_churn_bounded(&snapshots, 2, 0.15, 0.20, 0.20, 0.20);

    // Both topics should still be recallable independently under Limited mode
    let dark_results = aura.recall_structured(
        "dark mode code editor", Some(20), Some(0.0), Some(true), None, None,
    ).expect("recall dark");
    let light_results = aura.recall_structured(
        "light mode documentation reading", Some(20), Some(0.0), Some(true), None, None,
    ).expect("recall light");
    assert!(!dark_results.is_empty(), "dark mode should be recallable under Limited");
    assert!(!light_results.is_empty(), "light mode should be recallable under Limited");

    // Rerank comparison
    aura.set_belief_rerank_mode(BeliefRerankMode::Off); // reset before compare_rerank
    let comparisons: Vec<RerankComparison> = queries.iter()
        .map(|q| compare_rerank(&aura, q, 20))
        .collect();
    assert_rerank_bounded(&comparisons);

    let last = snapshots.last().unwrap();
    assert!(last.unresolved_beliefs <= last.belief_count, "unresolved ≤ total beliefs");
    assert!(last.policy_count <= 3, "policy minimal for preferences: got {}", last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Test D: Multi-Topic Mixed — topic isolation under rerank
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_scenario_d_multi_topic_with_rerank() {
    let queries = &[
        "deployment pipeline rollback",
        "UI design readability",
        "structured logging observability",
        "API security authentication",
    ];
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_d_records());

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for i in 0..8 {
        let report = aura.run_maintenance();
        snapshots.push(take_snapshot(&aura, i, &report));
        reports.push(report);
    }

    aura.set_belief_rerank_mode(BeliefRerankMode::Limited);

    assert_timings_valid(&reports);
    assert_no_provenance_gaps(&snapshots);
    assert_suppression_bounded(&snapshots);
    assert_churn_bounded(&snapshots, 2, 0.15, 0.25, 0.25, 0.25);

    // All 4 topics recallable under Limited mode
    for query in queries {
        let results = aura.recall_structured(query, Some(20), Some(0.0), Some(true), None, None)
            .expect("recall failed");
        assert!(!results.is_empty(), "topic '{}' recallable under Limited", query);
    }

    // Cross-topic isolation: concepts should not merge unrelated topics
    let concepts = aura.get_concepts(Some("stable"));
    for concept in &concepts {
        let has_deploy = concept.core_terms.iter().any(|t| t.contains("deploy"));
        let has_security = concept.core_terms.iter().any(|t| t.contains("security"));
        let has_ui = concept.core_terms.iter().any(|t| t.contains("ui") || t.contains("design"));
        let has_logging = concept.core_terms.iter().any(|t| t.contains("logging"));

        let topic_count = [has_deploy, has_security, has_ui, has_logging].iter()
            .filter(|&&x| x).count();
        assert!(topic_count <= 1,
            "concept '{}' merges {} topics (core_terms: {:?})", concept.key, topic_count, concept.core_terms);
    }

    // Rerank comparison across all 4 topics
    aura.set_belief_rerank_mode(BeliefRerankMode::Off);
    let comparisons: Vec<RerankComparison> = queries.iter()
        .map(|q| compare_rerank(&aura, q, 20))
        .collect();
    assert_rerank_bounded(&comparisons);

    let last = snapshots.last().unwrap();
    assert!(last.total_records >= 1, "records should survive");
    assert!(last.policy_count <= 10, "policy bounded: got {}", last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Soak: 20 cycles with Limited mode active from start
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_soak_20_cycles_with_rerank() {
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_b_records());

    // Enable Limited mode BEFORE maintenance cycles
    aura.set_belief_rerank_mode(BeliefRerankMode::Limited);

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();
    for i in 0..20 {
        let report = aura.run_maintenance();
        snapshots.push(take_snapshot(&aura, i, &report));
        reports.push(report);
    }

    assert_timings_valid(&reports);
    assert_no_provenance_gaps(&snapshots);
    assert_suppression_bounded(&snapshots);

    // Count stability in last 5 cycles
    let last_5 = &snapshots[15..];
    fn max_minus_min(v: &[usize]) -> usize {
        v.iter().max().unwrap_or(&0) - v.iter().min().unwrap_or(&0)
    }

    let belief_counts: Vec<usize> = last_5.iter().map(|s| s.belief_count).collect();
    let concept_counts: Vec<usize> = last_5.iter().map(|s| s.concept_count).collect();
    let causal_counts: Vec<usize> = last_5.iter().map(|s| s.causal_count).collect();
    let policy_counts: Vec<usize> = last_5.iter().map(|s| s.policy_count).collect();

    assert!(max_minus_min(&belief_counts) <= 2, "belief stabilize: {:?}", belief_counts);
    assert!(max_minus_min(&concept_counts) <= 2, "concept stabilize: {:?}", concept_counts);
    assert!(max_minus_min(&causal_counts) <= 2, "causal stabilize: {:?}", causal_counts);
    assert!(max_minus_min(&policy_counts) <= 2, "policy stabilize: {:?}", policy_counts);

    // Churn bounded in late cycles
    assert_churn_bounded(last_5, 0, 0.10, 0.15, 0.15, 0.15);

    // Rerank comparison at end of soak: Off vs Limited should be stable
    let queries = &["deploy staging canary", "hotfix production incident"];
    aura.set_belief_rerank_mode(BeliefRerankMode::Off);
    let comparisons: Vec<RerankComparison> = queries.iter()
        .map(|q| compare_rerank(&aura, q, 20))
        .collect();
    assert_rerank_bounded(&comparisons);
}

// ═══════════════════════════════════════════════════════════
// Aggregate: run all scenarios, collect summary metrics
// ═══════════════════════════════════════════════════════════

#[test]
fn step9_aggregate_rerank_cross_layer_summary() {
    // Run each scenario, collect rerank comparisons, print summary
    let scenarios: Vec<(&str, Vec<ScenarioRecord>, Vec<&str>)> = vec![
        ("A: Stable Preference", scenario_a_records(), vec!["dark mode editor preference", "Rust compiler tooling"]),
        ("B: Deploy Chain", scenario_b_records(), vec!["deploy staging canary", "hotfix production incident"]),
        ("C: Contextual Preference", scenario_c_records(), vec!["dark mode code editor", "light mode documentation reading"]),
        ("D: Multi-Topic Mixed", scenario_d_records(), vec![
            "deployment pipeline rollback", "UI design readability",
            "structured logging observability", "API security authentication",
        ]),
    ];

    let mut all_comparisons: Vec<RerankComparison> = Vec::new();
    let mut total_queries = 0;
    let mut reranked_count = 0;
    let mut moved_count = 0;

    for (name, records, queries) in &scenarios {
        let (aura, _dir) = open_temp_aura();
        store_scenario(&aura, records);
        for _ in 0..8 { aura.run_maintenance(); }

        for query in queries {
            let cmp = compare_rerank(&aura, query, 20);
            total_queries += 1;
            if cmp.report.was_applied { reranked_count += 1; }
            if cmp.report.records_moved > 0 { moved_count += 1; }
            all_comparisons.push(cmp);
        }

        // Print per-scenario summary
        eprintln!("  Scenario {}: {} queries processed", name, queries.len());
    }

    // Aggregate metrics
    let avg_overlap: f32 = all_comparisons.iter()
        .map(|c| c.top_k_overlap)
        .sum::<f32>() / all_comparisons.len() as f32;

    let max_up = all_comparisons.iter()
        .map(|c| c.report.max_up_shift)
        .max()
        .unwrap_or(0);
    let max_down = all_comparisons.iter()
        .map(|c| c.report.max_down_shift)
        .max()
        .unwrap_or(0);

    let avg_coverage: f32 = all_comparisons.iter()
        .map(|c| c.report.belief_coverage)
        .sum::<f32>() / all_comparisons.len() as f32;

    eprintln!("\n  === Step 9: Cross-Layer Rerank Summary ===");
    eprintln!("  Total queries:       {}", total_queries);
    eprintln!("  Reranked:            {}/{}", reranked_count, total_queries);
    eprintln!("  With movement:       {}/{}", moved_count, total_queries);
    eprintln!("  Avg top-k overlap:   {:.3}", avg_overlap);
    eprintln!("  Max up shift:        {}", max_up);
    eprintln!("  Max down shift:      {}", max_down);
    eprintln!("  Avg belief coverage: {:.3}", avg_coverage);

    // Safety gates
    assert!(avg_overlap >= 0.70, "avg overlap {} < 0.70", avg_overlap);
    assert!(max_up <= 2, "max up shift {} > 2", max_up);
    assert!(max_down <= 2, "max down shift {} > 2", max_down);
    assert_rerank_bounded(&all_comparisons);

    eprintln!("  Safety gates:        ALL PASS");
    eprintln!("  =====================================\n");
}
