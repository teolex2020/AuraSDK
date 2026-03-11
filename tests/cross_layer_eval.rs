//! Cross-layer evaluation harness for the full cognitive stack.
//!
//! Tests the complete pipeline end-to-end:
//!   Record -> Belief -> Concept -> Causal -> Policy
//!
//! Each scenario stores curated data, runs N maintenance cycles,
//! collects CycleSnapshots from MaintenanceReport + inspection helpers,
//! and asserts cross-layer invariants.
//!
//! Scenarios:
//!   A. Stable Preference Replay — stability + no noise
//!   B. Deploy/Safety Chain — causal + policy coherence
//!   C. Contextual Preference — no false conflict
//!   D. Multi-Topic Mixed Stream — topic isolation

use aura::{Aura, Level};
use aura::background_brain::MaintenanceReport;

// ═══════════════════════════════════════════════════════════
// Harness infrastructure
// ═══════════════════════════════════════════════════════════

/// A single record to store in the scenario.
struct ScenarioRecord {
    content: &'static str,
    level: Level,
    tags: &'static [&'static str],
    source_type: &'static str,
    semantic_type: &'static str,
}

/// Snapshot of one maintenance cycle — all layers.
#[derive(Debug)]
#[allow(dead_code)]
struct CycleSnapshot {
    cycle: usize,
    total_records: usize,

    // Belief layer
    belief_count: usize,
    resolved_beliefs: usize,
    unresolved_beliefs: usize,
    belief_churn: f32,

    // Concept layer
    concept_count: usize,
    stable_concepts: usize,
    concept_churn: f32,
    avg_concept_score: f32,

    // Causal layer
    causal_count: usize,
    stable_causal: usize,
    causal_churn: f32,
    avg_causal_score: f32,

    // Policy layer
    policy_count: usize,
    stable_policy: usize,
    suppressed_policy: usize,
    policy_churn: f32,
    avg_policy_score: f32,

    // Timing
    total_ms: f64,

    // Provenance checks (counts of entities missing provenance)
    causal_missing_provenance: usize,
    policy_missing_provenance: usize,
}

/// Full evaluation result for a scenario.
#[allow(dead_code)]
struct EvalResult {
    name: &'static str,
    reports: Vec<MaintenanceReport>,
    snapshots: Vec<CycleSnapshot>,
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

fn recall(aura: &Aura, query: &str) -> Vec<(f32, aura::Record)> {
    aura.recall_structured(query, Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed")
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

    let avg_concept_score = if concepts.is_empty() { 0.0 } else {
        concepts.iter().map(|c| c.abstraction_score).sum::<f32>() / concepts.len() as f32
    };
    let avg_causal_score = if causal.is_empty() { 0.0 } else {
        causal.iter().map(|p| p.causal_strength).sum::<f32>() / causal.len() as f32
    };
    let avg_policy_score = if policy.is_empty() { 0.0 } else {
        policy.iter().map(|h| h.policy_strength).sum::<f32>() / policy.len() as f32
    };

    // Provenance checks: stable causal/policy must have non-empty provenance
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
        avg_concept_score,
        causal_count: causal.len(),
        stable_causal,
        causal_churn: report.stability.causal_churn,
        avg_causal_score,
        policy_count: policy.len(),
        stable_policy,
        suppressed_policy,
        policy_churn: report.stability.policy_churn,
        avg_policy_score,
        total_ms: report.timings.total_ms,
        causal_missing_provenance,
        policy_missing_provenance,
    }
}

fn run_eval(name: &'static str, records: &[ScenarioRecord], cycles: usize) -> EvalResult {
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, records);

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();

    for i in 0..cycles {
        let report = aura.run_maintenance();
        let snapshot = take_snapshot(&aura, i, &report);
        reports.push(report);
        snapshots.push(snapshot);
    }

    EvalResult { name, reports, snapshots }
}

fn run_eval_with_recall(
    name: &'static str,
    records: &[ScenarioRecord],
    cycles: usize,
    queries: &[&str],
) -> (EvalResult, Vec<Vec<usize>>) {
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, records);

    // Before-recall snapshot
    let before_counts: Vec<usize> = queries.iter()
        .map(|q| recall(&aura, q).len())
        .collect();

    let mut reports = Vec::new();
    let mut snapshots = Vec::new();

    for i in 0..cycles {
        let report = aura.run_maintenance();
        let snapshot = take_snapshot(&aura, i, &report);
        reports.push(report);
        snapshots.push(snapshot);
    }

    // After-recall snapshot
    let after_counts: Vec<usize> = queries.iter()
        .map(|q| recall(&aura, q).len())
        .collect();

    let eval = EvalResult { name, reports, snapshots };
    (eval, vec![before_counts, after_counts])
}

// ── Global invariant helpers ──

/// Assert: after warmup_cycles, churn is bounded for all layers.
fn assert_churn_bounded(snapshots: &[CycleSnapshot], warmup: usize,
                         belief_max: f32, concept_max: f32,
                         causal_max: f32, policy_max: f32) {
    for snap in snapshots.iter().skip(warmup) {
        assert!(snap.belief_churn <= belief_max,
            "cycle {}: belief_churn {} > max {}",
            snap.cycle, snap.belief_churn, belief_max);
        assert!(snap.concept_churn <= concept_max,
            "cycle {}: concept_churn {} > max {}",
            snap.cycle, snap.concept_churn, concept_max);
        assert!(snap.causal_churn <= causal_max,
            "cycle {}: causal_churn {} > max {}",
            snap.cycle, snap.causal_churn, causal_max);
        assert!(snap.policy_churn <= policy_max,
            "cycle {}: policy_churn {} > max {}",
            snap.cycle, snap.policy_churn, policy_max);
    }
}

/// Assert: no provenance gaps on stable entities.
fn assert_no_provenance_gaps(snapshots: &[CycleSnapshot]) {
    for snap in snapshots {
        assert_eq!(snap.causal_missing_provenance, 0,
            "cycle {}: {} stable causal patterns missing provenance",
            snap.cycle, snap.causal_missing_provenance);
        assert_eq!(snap.policy_missing_provenance, 0,
            "cycle {}: {} stable policy hints missing provenance",
            snap.cycle, snap.policy_missing_provenance);
    }
}

/// Assert: timings are non-negative.
fn assert_timings_valid(reports: &[MaintenanceReport]) {
    for (i, r) in reports.iter().enumerate() {
        assert!(r.timings.total_ms >= 0.0,
            "cycle {}: negative total_ms {}", i, r.timings.total_ms);
        assert!(r.timings.belief_ms >= 0.0,
            "cycle {}: negative belief_ms {}", i, r.timings.belief_ms);
    }
}

/// Assert: recall does not degrade.
fn assert_recall_not_degraded(recall_counts: &[Vec<usize>]) {
    let before = &recall_counts[0];
    let after = &recall_counts[1];
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert!(*a >= *b,
            "query {}: recall degraded from {} to {}", i, b, a);
    }
}

/// Assert: suppressed count never exceeds total hints.
fn assert_suppression_bounded(snapshots: &[CycleSnapshot]) {
    for snap in snapshots {
        assert!(snap.suppressed_policy <= snap.policy_count,
            "cycle {}: suppressed {} > total hints {}",
            snap.cycle, snap.suppressed_policy, snap.policy_count);
    }
}

// ═══════════════════════════════════════════════════════════
// Scenario A: Stable Preference Replay
// ═══════════════════════════════════════════════════════════
//
// Goal: one stable topic with paraphrases. Expect low churn,
// stable beliefs dominating, no policy noise.

fn scenario_a_records() -> Vec<ScenarioRecord> {
    vec![
        ScenarioRecord { content: "I prefer dark mode in my IDE for coding", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark mode reduces eye strain during long coding sessions", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "My preferred editor theme is dark with high contrast", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark backgrounds work better for evening programming", level: Level::Domain, tags: &["editor", "preference"], source_type: "recorded", semantic_type: "fact" },
        // Nearby unrelated topic (should not merge)
        ScenarioRecord { content: "Rust compiler gives helpful error messages", level: Level::Domain, tags: &["rust", "tooling"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Cargo build system handles dependencies well", level: Level::Domain, tags: &["rust", "tooling"], source_type: "recorded", semantic_type: "fact" },
    ]
}

#[test]
fn scenario_a_stable_preference_replay() {
    let queries = &["dark mode editor preference", "Rust compiler tooling"];
    let (eval, recall_counts) = run_eval_with_recall(
        "stable_preference_replay", &scenario_a_records(), 8, queries,
    );

    // Global invariants
    assert_timings_valid(&eval.reports);
    assert_no_provenance_gaps(&eval.snapshots);
    assert_recall_not_degraded(&recall_counts);
    assert_suppression_bounded(&eval.snapshots);

    // Stability: after 2 warmup cycles, churn bounded
    assert_churn_bounded(&eval.snapshots, 2, 0.15, 0.20, 0.20, 0.20);

    let last = eval.snapshots.last().unwrap();

    // If beliefs formed, unresolved should not exceed total
    if last.belief_count > 0 {
        assert!(last.unresolved_beliefs <= last.belief_count,
            "unresolved should not exceed total beliefs: {} unresolved out of {}",
            last.unresolved_beliefs, last.belief_count);
    }

    // Records should survive
    assert!(last.total_records >= 1, "records should survive maintenance");

    // Policy hints should be bounded (no strong causal signal in preferences)
    assert!(last.policy_count <= 5,
        "policy count should be bounded for stable preferences: got {}",
        last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Scenario B: Deploy/Safety Chain
// ═══════════════════════════════════════════════════════════
//
// Goal: causal chain from deploy decisions to outcomes.
// Expect concepts, causal patterns, and possibly policy hints.

fn scenario_b_records() -> Vec<ScenarioRecord> {
    vec![
        // Deploy chain 1: successful
        ScenarioRecord { content: "Deployed canary release to staging before production", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Canary testing caught no regressions in staging", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Production deploy succeeded after canary approval", level: Level::Domain, tags: &["deploy", "production"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Canary release process improved deployment safety", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        // Deploy chain 2: incident
        ScenarioRecord { content: "Pushed hotfix directly to production without staging", level: Level::Domain, tags: &["deploy", "production", "hotfix"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Hotfix caused unexpected error rate spike in production", level: Level::Domain, tags: &["deploy", "production", "incident"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Emergency rollback was required after hotfix failure", level: Level::Domain, tags: &["deploy", "production", "incident"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Post-incident review recommended mandatory staging", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "decision" },
        // Supporting evidence
        ScenarioRecord { content: "Staging environments reduce production incident rate", level: Level::Domain, tags: &["deploy", "staging"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Deploy pipeline includes automated smoke tests", level: Level::Domain, tags: &["deploy", "testing"], source_type: "recorded", semantic_type: "fact" },
    ]
}

#[test]
fn scenario_b_deploy_safety_chain() {
    let queries = &["deploy staging canary", "hotfix production incident"];
    let (eval, recall_counts) = run_eval_with_recall(
        "deploy_safety_chain", &scenario_b_records(), 8, queries,
    );

    // Global invariants
    assert_timings_valid(&eval.reports);
    assert_no_provenance_gaps(&eval.snapshots);
    assert_recall_not_degraded(&recall_counts);
    assert_suppression_bounded(&eval.snapshots);

    // Stability after warmup
    assert_churn_bounded(&eval.snapshots, 3, 0.15, 0.25, 0.25, 0.25);

    let last = eval.snapshots.last().unwrap();

    // If beliefs formed, check consistency
    if last.belief_count > 0 {
        assert!(last.unresolved_beliefs <= last.belief_count,
            "unresolved should not exceed total beliefs");
    }

    // Records survive
    assert!(last.total_records >= 1, "records should survive");

    // Policy hints should be bounded (may or may not form depending on causal strength)
    assert!(last.policy_count <= 10,
        "policy count should be bounded: got {}", last.policy_count);

    // If policy hints exist, suppressed should not exceed half
    if last.policy_count > 0 {
        assert!(last.suppressed_policy <= last.policy_count / 2 + 1,
            "suppression should not dominate: {} suppressed of {} total",
            last.suppressed_policy, last.policy_count);
    }
}

// ═══════════════════════════════════════════════════════════
// Scenario C: Contextual Preference
// ═══════════════════════════════════════════════════════════
//
// Goal: similar topics in different contexts should not
// create false conflicts or false abstractions.

fn scenario_c_records() -> Vec<ScenarioRecord> {
    vec![
        // Dark mode for editor
        ScenarioRecord { content: "I use dark mode in my code editor for comfort", level: Level::Domain, tags: &["editor", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Dark editor themes help focus during programming", level: Level::Domain, tags: &["editor", "theme"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "VS Code dark theme is my primary editor setting", level: Level::Domain, tags: &["editor", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        // Light mode for docs
        ScenarioRecord { content: "I prefer light mode when reading documentation", level: Level::Domain, tags: &["docs", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
        ScenarioRecord { content: "Light backgrounds improve readability for long documents", level: Level::Domain, tags: &["docs", "theme"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Documentation sites look better with light themes", level: Level::Domain, tags: &["docs", "theme", "preference"], source_type: "recorded", semantic_type: "preference" },
    ]
}

#[test]
fn scenario_c_contextual_preference() {
    let queries = &["dark mode editor", "light mode documentation"];
    let (eval, recall_counts) = run_eval_with_recall(
        "contextual_preference", &scenario_c_records(), 8, queries,
    );

    // Global invariants
    assert_timings_valid(&eval.reports);
    assert_no_provenance_gaps(&eval.snapshots);
    assert_recall_not_degraded(&recall_counts);
    assert_suppression_bounded(&eval.snapshots);

    // Stability
    assert_churn_bounded(&eval.snapshots, 2, 0.15, 0.20, 0.20, 0.20);

    let last = eval.snapshots.last().unwrap();

    // Both topics should be recallable independently
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_c_records());
    for _ in 0..5 { aura.run_maintenance(); }

    let dark_results = recall(&aura, "dark mode code editor");
    let light_results = recall(&aura, "light mode documentation reading");
    assert!(!dark_results.is_empty(), "dark mode topic should be recallable");
    assert!(!light_results.is_empty(), "light mode topic should be recallable");

    // Unresolved beliefs should not explode
    assert!(last.unresolved_beliefs <= last.belief_count,
        "unresolved should not exceed total beliefs");

    // Policy hints should be absent or very weak (no causal signal)
    assert!(last.policy_count <= 3,
        "policy should be minimal for preference data: got {}", last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Scenario D: Multi-Topic Mixed Stream
// ═══════════════════════════════════════════════════════════
//
// Goal: 4 distinct topics in same namespace should stay
// isolated across all layers.

fn scenario_d_records() -> Vec<ScenarioRecord> {
    vec![
        // Topic 1: Deploy
        ScenarioRecord { content: "Blue-green deployment minimizes downtime", level: Level::Domain, tags: &["deploy", "infrastructure"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Automated rollback triggers on error rate threshold", level: Level::Domain, tags: &["deploy", "infrastructure"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Deploy pipeline runs integration tests before release", level: Level::Domain, tags: &["deploy", "testing"], source_type: "recorded", semantic_type: "decision" },
        // Topic 2: UI preferences
        ScenarioRecord { content: "Sans-serif fonts improve UI readability on screens", level: Level::Domain, tags: &["ui", "design"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Consistent spacing in UI components reduces cognitive load", level: Level::Domain, tags: &["ui", "design"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Accessible color contrast ratios are mandatory", level: Level::Domain, tags: &["ui", "accessibility"], source_type: "recorded", semantic_type: "decision" },
        // Topic 3: Logging
        ScenarioRecord { content: "Structured JSON logs enable efficient aggregation", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Log levels should follow severity conventions", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Centralized logging reduces debugging time", level: Level::Domain, tags: &["logging", "observability"], source_type: "recorded", semantic_type: "fact" },
        // Topic 4: Security
        ScenarioRecord { content: "All API endpoints require authentication tokens", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "decision" },
        ScenarioRecord { content: "Rate limiting prevents abuse of public endpoints", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "fact" },
        ScenarioRecord { content: "Security headers configured on all HTTP responses", level: Level::Domain, tags: &["security", "api"], source_type: "recorded", semantic_type: "decision" },
    ]
}

#[test]
fn scenario_d_multi_topic_mixed_stream() {
    let queries = &[
        "deployment pipeline rollback",
        "UI design readability",
        "structured logging observability",
        "API security authentication",
    ];
    let (eval, recall_counts) = run_eval_with_recall(
        "multi_topic_mixed_stream", &scenario_d_records(), 8, queries,
    );

    // Global invariants
    assert_timings_valid(&eval.reports);
    assert_no_provenance_gaps(&eval.snapshots);
    assert_recall_not_degraded(&recall_counts);
    assert_suppression_bounded(&eval.snapshots);

    // Stability
    assert_churn_bounded(&eval.snapshots, 2, 0.15, 0.25, 0.25, 0.25);

    let last = eval.snapshots.last().unwrap();

    // All 4 queries should return results
    let (aura, _dir) = open_temp_aura();
    store_scenario(&aura, &scenario_d_records());
    for _ in 0..5 { aura.run_maintenance(); }

    for query in queries {
        let results = recall(&aura, query);
        assert!(!results.is_empty(),
            "topic '{}' should be recallable after maintenance", query);
    }

    // Cross-topic isolation: check concepts don't merge across topics
    let concepts = aura.get_concepts(Some("stable"));
    for concept in &concepts {
        // A stable concept should not have tags from multiple unrelated topics.
        // Check that it doesn't simultaneously contain deploy + security tags.
        let has_deploy = concept.core_terms.iter().any(|t| t.contains("deploy"));
        let has_security = concept.core_terms.iter().any(|t| t.contains("security"));
        let has_ui = concept.core_terms.iter().any(|t| t.contains("ui") || t.contains("design"));
        let has_logging = concept.core_terms.iter().any(|t| t.contains("logging"));

        let topic_count = [has_deploy, has_security, has_ui, has_logging].iter()
            .filter(|&&x| x).count();
        assert!(topic_count <= 1,
            "concept '{}' merges {} distinct topics (core_terms: {:?})",
            concept.key, topic_count, concept.core_terms);
    }

    // Records survive
    assert!(last.total_records >= 1, "records should survive");

    // Counts bounded
    assert!(last.policy_count <= 10,
        "policy count bounded for mixed stream: got {}", last.policy_count);
}

// ═══════════════════════════════════════════════════════════
// Soak: 20 cycles, verify no divergence
// ═══════════════════════════════════════════════════════════

#[test]
fn soak_cross_layer_20_cycles() {
    let eval = run_eval("soak_20_cycles", &scenario_b_records(), 20);

    // Timings valid
    assert_timings_valid(&eval.reports);
    assert_no_provenance_gaps(&eval.snapshots);
    assert_suppression_bounded(&eval.snapshots);

    // After warmup, counts should be stable (no runaway growth)
    let last_5 = &eval.snapshots[15..];

    let belief_counts: Vec<usize> = last_5.iter().map(|s| s.belief_count).collect();
    let concept_counts: Vec<usize> = last_5.iter().map(|s| s.concept_count).collect();
    let causal_counts: Vec<usize> = last_5.iter().map(|s| s.causal_count).collect();
    let policy_counts: Vec<usize> = last_5.iter().map(|s| s.policy_count).collect();

    fn max_minus_min(v: &[usize]) -> usize {
        v.iter().max().unwrap_or(&0) - v.iter().min().unwrap_or(&0)
    }

    assert!(max_minus_min(&belief_counts) <= 2,
        "belief count should stabilize: {:?}", belief_counts);
    assert!(max_minus_min(&concept_counts) <= 2,
        "concept count should stabilize: {:?}", concept_counts);
    assert!(max_minus_min(&causal_counts) <= 2,
        "causal count should stabilize: {:?}", causal_counts);
    assert!(max_minus_min(&policy_counts) <= 2,
        "policy count should stabilize: {:?}", policy_counts);

    // Churn bounded in late cycles
    assert_churn_bounded(last_5, 0, 0.10, 0.15, 0.15, 0.15);
}
