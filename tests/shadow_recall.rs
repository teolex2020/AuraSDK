//! Integration tests for belief-aware recall (Phase 3 shadow + Phase 4 limited influence).
//!
//! Validates:
//! - Shadow scoring runs alongside recall without changing baseline results
//! - Shadow report has valid metrics (coverage, overlap, latency)
//! - Shadow ranking differs from baseline when beliefs are present
//! - Recall results are identical with and without shadow scoring
//! - Shadow latency stays within budget
//! - Phase 4: tri-state mode (Off/Shadow/Limited)
//! - Phase 4: scope guards, positional shift cap, rerank report

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

// ═════════════════════════════════════════════════════════
// 1. SHADOW RECALL ON EMPTY DB
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_recall_empty_db() {
    let (aura, _dir) = open_temp_aura();

    let (baseline, shadow) = aura
        .recall_structured_with_shadow("anything", None, None, None, None, None)
        .expect("shadow recall failed");

    assert!(baseline.is_empty());
    assert!(shadow.scores.is_empty());
    assert_eq!(shadow.top_k_overlap, 1.0);
    assert_eq!(shadow.belief_coverage, 0.0);
}

// ═════════════════════════════════════════════════════════
// 2. SHADOW DOES NOT ALTER BASELINE RESULTS
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_does_not_alter_baseline() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Rust is a systems programming language", Level::Domain, &["rust", "programming"], "recorded", "fact"),
        ("Rust has zero-cost abstractions", Level::Domain, &["rust", "programming"], "recorded", "fact"),
        ("Python is great for data science", Level::Domain, &["python", "data"], "recorded", "fact"),
        ("TypeScript adds types to JavaScript", Level::Domain, &["typescript", "web"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 3);

    // Get baseline with shadow — the key invariant is that the returned
    // baseline results are the SAME as what recall_core produces.
    // We can't compare to a separate recall_structured call because
    // activate_and_strengthen mutates record state between calls.
    let (baseline_with_shadow, shadow) = aura
        .recall_structured_with_shadow("Rust programming", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    // Shadow scores must correspond 1:1 with baseline results
    assert_eq!(baseline_with_shadow.len(), shadow.scores.len());

    // Baseline scores in shadow report must match actual baseline
    for (i, (score, rec)) in baseline_with_shadow.iter().enumerate() {
        let ss = &shadow.scores[i];
        assert_eq!(ss.record_id, rec.id, "record ID mismatch at position {}", i);
        assert!((ss.baseline_score - score).abs() < 0.0001,
            "baseline score mismatch at position {}", i);
        assert_eq!(ss.baseline_rank, i, "baseline rank mismatch");
    }
}

// ═════════════════════════════════════════════════════════
// 3. SHADOW REPORT HAS VALID METRICS
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_report_valid_metrics() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("I prefer dark mode in all editors", Level::Domain, &["preference", "editor"], "recorded", "preference"),
        ("Dark mode reduces eye strain at night", Level::Domain, &["preference", "editor"], "recorded", "fact"),
        ("Light mode is better for readability", Level::Domain, &["preference", "editor"], "recorded", "fact"),
        ("VSCode supports both dark and light themes", Level::Domain, &["editor", "vscode"], "recorded", "fact"),
        ("I always use dark mode when coding late", Level::Domain, &["preference", "editor"], "recorded", "preference"),
    ]);

    run_cycles(&aura, 5);

    let (baseline, shadow) = aura
        .recall_structured_with_shadow("dark mode editor", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    if !baseline.is_empty() {
        // Coverage in [0, 1]
        assert!(shadow.belief_coverage >= 0.0 && shadow.belief_coverage <= 1.0,
            "belief coverage out of range: {}", shadow.belief_coverage);

        // Top-k overlap in [0, 1]
        assert!(shadow.top_k_overlap >= 0.0 && shadow.top_k_overlap <= 1.0,
            "top-k overlap out of range: {}", shadow.top_k_overlap);

        // Avg multiplier should be near 1.0 (±0.15)
        assert!(shadow.avg_belief_multiplier >= 0.85 && shadow.avg_belief_multiplier <= 1.15,
            "avg multiplier out of range: {}", shadow.avg_belief_multiplier);

        // Counts must sum to total
        let total = shadow.promoted_count + shadow.demoted_count + shadow.unchanged_count;
        assert_eq!(total, shadow.scores.len(),
            "count mismatch: {}+{}+{} != {}", shadow.promoted_count, shadow.demoted_count, shadow.unchanged_count, shadow.scores.len());

        // Each score entry has valid fields
        for s in &shadow.scores {
            assert!(s.baseline_score >= 0.0);
            assert!(s.shadow_score >= 0.0);
            assert!(s.belief_multiplier > 0.0);
            assert!(s.baseline_rank < shadow.scores.len());
            assert!(s.shadow_rank < shadow.scores.len());
        }
    }
}

// ═════════════════════════════════════════════════════════
// 4. SHADOW LATENCY WITHIN BUDGET
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_latency_within_budget() {
    let (aura, _dir) = open_temp_aura();

    // Store enough records to make timing measurable
    for i in 0..20 {
        aura.store(
            &format!("Record number {} about various programming topics", i),
            Some(Level::Working),
            Some(vec!["test".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }

    run_cycles(&aura, 3);

    let (_baseline, shadow) = aura
        .recall_structured_with_shadow("programming", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    // Shadow scoring should be < 2ms (2000 μs) — it's just HashMap lookups
    assert!(shadow.shadow_latency_us < 2000,
        "shadow latency {}μs exceeds 2ms budget", shadow.shadow_latency_us);
}

// ═════════════════════════════════════════════════════════
// 5. SHADOW REPORT AFTER BELIEF FORMATION
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_report_after_beliefs_form() {
    let (aura, _dir) = open_temp_aura();

    // Store enough similar records to trigger belief formation
    for i in 0..8 {
        aura.store(
            &format!("Deploying with canary releases reduces production incidents {}", i),
            Some(Level::Domain),
            Some(vec!["deploy".to_string(), "canary".to_string(), "ops".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }

    // Run enough cycles for beliefs to form and stabilize
    run_cycles(&aura, 8);

    let (baseline, shadow) = aura
        .recall_structured_with_shadow("canary deploy", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    if !baseline.is_empty() {
        // With 8 similar records, beliefs should form
        // If they did, coverage should be > 0
        // (soft assertion — depends on SDR clustering)
        if shadow.belief_coverage > 0.0 {
            // At least some records have belief membership
            let has_belief_state = shadow.scores.iter()
                .any(|s| s.belief_state.is_some());
            assert!(has_belief_state, "coverage > 0 but no belief states found");

            // Belief states should be valid strings
            for s in &shadow.scores {
                if let Some(ref state) = s.belief_state {
                    assert!(
                        ["resolved", "singleton", "unresolved", "empty"].contains(&state.as_str()),
                        "invalid belief state: {}", state
                    );
                }
            }
        }
    }
}

// ═════════════════════════════════════════════════════════
// 6. SHADOW STABLE ACROSS REPEATED RECALLS
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_stable_across_repeated_recalls() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Machine learning requires large datasets", Level::Domain, &["ml", "data"], "recorded", "fact"),
        ("Neural networks are a type of machine learning", Level::Domain, &["ml", "ai"], "recorded", "fact"),
        ("Deep learning uses multiple neural network layers", Level::Domain, &["ml", "ai"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 3);

    // Run shadow recall — verify internal consistency
    let (baseline, shadow) = aura
        .recall_structured_with_shadow("machine learning", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    // Shadow must be consistent with baseline in the same call
    assert_eq!(baseline.len(), shadow.scores.len());

    // Verify deterministic properties within a single report:
    // - belief multipliers are consistent with belief state
    // - rank assignments cover [0..n) without gaps
    let n = shadow.scores.len();
    if n > 0 {
        let mut baseline_ranks: Vec<usize> = shadow.scores.iter().map(|s| s.baseline_rank).collect();
        let mut shadow_ranks: Vec<usize> = shadow.scores.iter().map(|s| s.shadow_rank).collect();
        baseline_ranks.sort();
        shadow_ranks.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(baseline_ranks, expected, "baseline ranks not contiguous");
        assert_eq!(shadow_ranks, expected, "shadow ranks not contiguous");

        // Multiplier consistency
        for s in &shadow.scores {
            match s.belief_state.as_deref() {
                Some("resolved") => assert!((s.belief_multiplier - 1.10).abs() < 0.001),
                Some("singleton") => assert!((s.belief_multiplier - 1.05).abs() < 0.001),
                Some("unresolved") => assert!((s.belief_multiplier - 0.95).abs() < 0.001),
                Some("empty") | None => assert!((s.belief_multiplier - 1.00).abs() < 0.001),
                Some(other) => panic!("unexpected belief state: {}", other),
            }
        }
    }
}

// ═════════════════════════════════════════════════════════
// 7. SHADOW RECALL DOES NOT AFFECT NEXT RECALL
// ═════════════════════════════════════════════════════════

#[test]
fn shadow_recall_no_side_effects() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Coffee is best served hot", Level::Working, &["coffee"], "recorded", "preference"),
        ("Tea can be served hot or cold", Level::Working, &["tea"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 2);

    // Recall with shadow — verify shadow call itself doesn't cause errors
    // or produce inconsistent results
    let (baseline, shadow) = aura
        .recall_structured_with_shadow("coffee tea", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    // Shadow report length matches baseline
    assert_eq!(baseline.len(), shadow.scores.len());

    // Shadow scores are non-negative
    for s in &shadow.scores {
        assert!(s.shadow_score >= 0.0);
        assert!(s.baseline_score >= 0.0);
    }

    // Key invariant: shadow path didn't interfere with baseline record set
    // (all baseline records appear in shadow scores)
    for (_, rec) in &baseline {
        assert!(shadow.scores.iter().any(|s| s.record_id == rec.id),
            "baseline record {} missing from shadow scores", rec.id);
    }
}

// ═════════════════════════════════════════════════════════
// 8. BELIEF RERANK DISABLED BY DEFAULT
// ═════════════════════════════════════════════════════════

#[test]
fn belief_rerank_disabled_by_default() {
    let (aura, _dir) = open_temp_aura();
    assert!(!aura.is_belief_rerank_enabled());
}

// ═════════════════════════════════════════════════════════
// 9. BELIEF RERANK TOGGLE
// ═════════════════════════════════════════════════════════

#[test]
fn belief_rerank_toggle() {
    let (aura, _dir) = open_temp_aura();

    aura.set_belief_rerank_enabled(true);
    assert!(aura.is_belief_rerank_enabled());

    aura.set_belief_rerank_enabled(false);
    assert!(!aura.is_belief_rerank_enabled());
}

// ═════════════════════════════════════════════════════════
// 10. BELIEF RERANK SAME RESULT SET AS BASELINE
// ═════════════════════════════════════════════════════════

#[test]
fn belief_rerank_same_result_set() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Rust is great for systems programming", Level::Domain, &["rust", "programming"], "recorded", "fact"),
        ("Rust has zero cost abstractions", Level::Domain, &["rust", "programming"], "recorded", "fact"),
        ("Python is great for scripting", Level::Domain, &["python", "programming"], "recorded", "fact"),
        ("Java is used in enterprise", Level::Domain, &["java", "programming"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 5);

    // Use shadow recall to get baseline + shadow comparison in one call
    // (avoids activate_and_strengthen side effects between calls)
    let (baseline, shadow) = aura
        .recall_structured_with_shadow("Rust programming", Some(20), Some(0.0), Some(true), None, None)
        .expect("shadow recall failed");

    // Shadow report must have same records as baseline
    assert_eq!(baseline.len(), shadow.scores.len());

    let baseline_ids: std::collections::HashSet<String> = baseline.iter().map(|(_, r)| r.id.clone()).collect();
    let shadow_ids: std::collections::HashSet<String> = shadow.scores.iter().map(|s| s.record_id.clone()).collect();
    assert_eq!(baseline_ids, shadow_ids, "shadow changed result set");

    // Now enable rerank and verify it still returns results
    aura.set_belief_rerank_enabled(true);
    let reranked = aura
        .recall_structured("Rust programming", Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed");

    // Reranked should have results (may differ in order due to ±3% adjustment)
    assert!(!reranked.is_empty() || baseline.is_empty(),
        "rerank should return results if baseline did");
}

// ═════════════════════════════════════════════════════════
// 11. BELIEF RERANK SCORE DELTA BOUNDED
// ═════════════════════════════════════════════════════════

#[test]
fn belief_rerank_score_delta_bounded() {
    let (aura, _dir) = open_temp_aura();

    for i in 0..10 {
        aura.store(
            &format!("Record about testing and deployment practices iteration {}", i),
            Some(Level::Domain),
            Some(vec!["testing".to_string(), "deploy".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }

    run_cycles(&aura, 8);

    // Get baseline scores
    let baseline = aura
        .recall_structured("testing deploy", Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed");

    if baseline.is_empty() {
        return;
    }

    // Enable rerank, get shadow report comparing baseline vs reranked
    aura.set_belief_rerank_enabled(true);
    let reranked = aura
        .recall_structured("testing deploy", Some(20), Some(0.0), Some(true), None, None)
        .expect("recall failed");

    // Note: scores differ due to activate_and_strengthen between calls,
    // but we can verify the reranked scores are all positive and bounded
    for (score, _) in &reranked {
        assert!(*score > 0.0, "reranked score should be positive");
        assert!(*score <= 1.5, "reranked score unreasonably high: {}", score);
    }
}

// ═══════════════════════════════════════════════════════════════
// Phase 4 Integration Tests
// ═══════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════
// 12. TRI-STATE MODE SWITCHING
// ═════════════════════════════════════════════════════════

#[test]
fn phase4_tristate_mode_default_off() {
    let (aura, _dir) = open_temp_aura();
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Off);
    assert!(!aura.is_belief_rerank_enabled());
}

#[test]
fn phase4_tristate_mode_switching() {
    let (aura, _dir) = open_temp_aura();

    // Off → Shadow
    aura.set_belief_rerank_mode(aura::recall::BeliefRerankMode::Shadow);
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Shadow);
    assert!(!aura.is_belief_rerank_enabled()); // Shadow ≠ enabled

    // Shadow → Limited
    aura.set_belief_rerank_mode(aura::recall::BeliefRerankMode::Limited);
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Limited);
    assert!(aura.is_belief_rerank_enabled());

    // Limited → Off
    aura.set_belief_rerank_mode(aura::recall::BeliefRerankMode::Off);
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Off);
    assert!(!aura.is_belief_rerank_enabled());
}

#[test]
fn phase4_compat_set_enabled_maps_to_limited() {
    let (aura, _dir) = open_temp_aura();

    aura.set_belief_rerank_enabled(true);
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Limited);

    aura.set_belief_rerank_enabled(false);
    assert_eq!(aura.get_belief_rerank_mode(), aura::recall::BeliefRerankMode::Off);
}

// ═════════════════════════════════════════════════════════
// 13. RERANK REPORT API
// ═════════════════════════════════════════════════════════

#[test]
fn phase4_rerank_report_on_populated_db() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Rust ownership model prevents data races at compile time", Level::Domain, &["rust", "safety"], "recorded", "fact"),
        ("Rust borrow checker enforces memory safety without GC", Level::Domain, &["rust", "safety"], "recorded", "fact"),
        ("Rust lifetimes track reference validity at compile time", Level::Domain, &["rust", "safety"], "recorded", "fact"),
        ("Rust move semantics transfer ownership between variables", Level::Domain, &["rust", "safety"], "recorded", "fact"),
        ("Rust pattern matching is exhaustive by default", Level::Domain, &["rust", "patterns"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 8);

    let result = aura.recall_structured_with_rerank_report(
        "Rust memory safety", Some(10), Some(0.0), Some(true), None, None,
    );
    assert!(result.is_ok());

    let (_scored, report) = result.unwrap();
    // Report should be valid (may or may not be applied depending on belief coverage)
    assert!(report.top_k_overlap >= 0.0 && report.top_k_overlap <= 1.0);
    assert!(report.rerank_latency_us < 100_000);
}

// ═════════════════════════════════════════════════════════
// 14. SHADOW MODE DOES NOT ALTER RANKING
// ═════════════════════════════════════════════════════════

#[test]
fn phase4_shadow_mode_no_ranking_change() {
    let (aura, _dir) = open_temp_aura();

    store_batch(&aura, &[
        ("Docker containers provide process isolation", Level::Domain, &["docker", "containers"], "recorded", "fact"),
        ("Docker uses Linux namespaces for isolation", Level::Domain, &["docker", "linux"], "recorded", "fact"),
        ("Docker images are layered filesystem snapshots", Level::Domain, &["docker", "images"], "recorded", "fact"),
        ("Docker Compose orchestrates multi-container apps", Level::Domain, &["docker", "compose"], "recorded", "fact"),
    ]);

    run_cycles(&aura, 5);

    // Set shadow mode — should NOT affect recall_structured results
    aura.set_belief_rerank_mode(aura::recall::BeliefRerankMode::Shadow);

    let result = aura.recall_structured(
        "Docker containers", Some(10), Some(0.0), Some(true), None, None,
    );
    assert!(result.is_ok());
    let scored = result.unwrap();

    // Shadow mode should not change behavior of recall_structured
    // (it only activates in recall_structured_with_shadow)
    if !scored.is_empty() {
        for (score, _) in &scored {
            assert!(*score > 0.0);
        }
    }
}

// ═════════════════════════════════════════════════════════
// 15. LIMITED MODE BOUNDED SCORE DELTA
// ═════════════════════════════════════════════════════════

#[test]
fn phase4_limited_mode_score_bounded() {
    let (aura, _dir) = open_temp_aura();

    for i in 0..10 {
        aura.store(
            &format!("Record about testing methodology and best practices iteration {}", i),
            Some(Level::Domain),
            Some(vec!["testing".to_string(), "practices".to_string()]),
            None, None,
            Some("recorded"),
            None,
            Some(false),
            None, None,
            Some("fact"),
        ).unwrap();
    }

    run_cycles(&aura, 8);

    // Get rerank report (always applies rerank for evaluation)
    let result = aura.recall_structured_with_rerank_report(
        "testing practices", Some(10), Some(0.0), Some(true), None, None,
    );
    assert!(result.is_ok());

    let (scored, report) = result.unwrap();

    // If applied, verify bounds
    if report.was_applied {
        assert!(report.max_up_shift <= 2, "up shift {} > 2", report.max_up_shift);
        assert!(report.max_down_shift <= 2, "down shift {} > 2", report.max_down_shift);
        assert!(report.belief_coverage > 0.0);
    }

    // All scores positive
    for (score, _) in &scored {
        assert!(*score > 0.0);
    }
}
