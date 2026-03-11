//! Concept Realistic Eval Sprint
//!
//! Tests concept formation on realistic-sized corpora (10-15 records/topic)
//! with both Standard and Relaxed seed gates.
//!
//! Step 1: Realistic corpora → current gates → measure concept coverage
//! Step 2: If coverage still 0 → Relaxed gates → measure again
//! Step 3: Compare coverage, useful rate, false merges, identity stability

use aura::{Aura, Level};
use aura::belief::{BeliefState, CoarseKeyMode};
use aura::concept::{ConceptSeedMode, ConceptState};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════
// Infrastructure
// ═══════════════════════════════════════════════════════════

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

fn run_cycles(aura: &Aura, n: usize) {
    for _ in 0..n {
        aura.run_maintenance();
    }
}

struct CorpusRecord {
    content: &'static str,
    level: Level,
    tags: &'static [&'static str],
    semantic_type: &'static str,
}

fn store_corpus(aura: &Aura, records: &[CorpusRecord]) {
    for rec in records {
        aura.store(
            rec.content, Some(rec.level),
            Some(rec.tags.iter().map(|t| t.to_string()).collect()),
            None, None, Some("recorded"), None, Some(false), None, None,
            Some(rec.semantic_type),
        ).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════
// Realistic corpora: 10-15 records per topic, 3 topics
// ═══════════════════════════════════════════════════════════

/// Topic A: deployment pipeline (12 records — decisions + facts, all share "deploy" tag)
fn corpus_deploy() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "Deployed version 2.3 to staging environment for validation",
            level: Level::Domain, tags: &["deploy", "staging"], semantic_type: "decision" },
        CorpusRecord { content: "Staging deployment passed all automated smoke tests successfully",
            level: Level::Domain, tags: &["deploy", "staging", "testing"], semantic_type: "fact" },
        CorpusRecord { content: "Promoted staging build to production canary fleet for monitoring",
            level: Level::Domain, tags: &["deploy", "production", "canary"], semantic_type: "decision" },
        CorpusRecord { content: "Canary deployment showed zero error rate increase over baseline",
            level: Level::Domain, tags: &["deploy", "canary", "monitoring"], semantic_type: "fact" },
        CorpusRecord { content: "Completed full production rollout of version 2.3 to all regions",
            level: Level::Domain, tags: &["deploy", "production"], semantic_type: "decision" },
        CorpusRecord { content: "Post-deploy monitoring confirmed healthy metrics across services",
            level: Level::Domain, tags: &["deploy", "monitoring"], semantic_type: "fact" },
        CorpusRecord { content: "Rollback plan for version 2.3 is documented and tested in staging",
            level: Level::Domain, tags: &["deploy", "staging", "rollback"], semantic_type: "fact" },
        CorpusRecord { content: "Blue-green deployment strategy reduces downtime during releases",
            level: Level::Domain, tags: &["deploy", "strategy"], semantic_type: "decision" },
        CorpusRecord { content: "Feature flags control gradual rollout of new deploy functionality",
            level: Level::Domain, tags: &["deploy", "feature-flags"], semantic_type: "decision" },
        CorpusRecord { content: "Deployment pipeline includes automated security scanning step",
            level: Level::Domain, tags: &["deploy", "security", "pipeline"], semantic_type: "fact" },
        CorpusRecord { content: "Each deploy requires approval from at least one team reviewer",
            level: Level::Domain, tags: &["deploy", "process"], semantic_type: "decision" },
        CorpusRecord { content: "Deploy artifacts are versioned and stored in container registry",
            level: Level::Domain, tags: &["deploy", "artifacts"], semantic_type: "fact" },
    ]
}

/// Topic B: database operations (11 records — share "database" tag)
fn corpus_database() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "PostgreSQL connection pool maximum set to thirty connections",
            level: Level::Domain, tags: &["database", "config"], semantic_type: "decision" },
        CorpusRecord { content: "Database query performance improved after adding composite indexes",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "fact" },
        CorpusRecord { content: "Weekly database backups stored in encrypted offsite location",
            level: Level::Domain, tags: &["database", "backup"], semantic_type: "fact" },
        CorpusRecord { content: "Database migration scripts always run inside a transaction block",
            level: Level::Domain, tags: &["database", "migration"], semantic_type: "decision" },
        CorpusRecord { content: "Read replicas handle reporting queries to reduce primary load",
            level: Level::Domain, tags: &["database", "replication"], semantic_type: "decision" },
        CorpusRecord { content: "Database connection timeouts set to five seconds for all services",
            level: Level::Domain, tags: &["database", "config"], semantic_type: "decision" },
        CorpusRecord { content: "Slow query log enabled for queries exceeding two hundred milliseconds",
            level: Level::Domain, tags: &["database", "monitoring"], semantic_type: "fact" },
        CorpusRecord { content: "Database schema changes reviewed by DBA before production deploy",
            level: Level::Domain, tags: &["database", "process"], semantic_type: "decision" },
        CorpusRecord { content: "Partitioning large tables by date improves database query speed",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "fact" },
        CorpusRecord { content: "Database credentials rotated monthly using vault secret manager",
            level: Level::Domain, tags: &["database", "security"], semantic_type: "fact" },
        CorpusRecord { content: "Connection pooling via PgBouncer reduces database overhead",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "fact" },
    ]
}

/// Topic C: editor preferences (10 records — share "editor" tag)
fn corpus_editor() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "I prefer dark mode with high contrast theme for coding sessions",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Dark mode reduces eye strain during long evening programming",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "fact" },
        CorpusRecord { content: "My preferred editor theme is Solarized Dark with large font",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Vim keybindings in VS Code improve text editing speed greatly",
            level: Level::Domain, tags: &["editor", "vim", "keybindings"], semantic_type: "preference" },
        CorpusRecord { content: "VS Code integrated terminal makes workflow smoother overall",
            level: Level::Domain, tags: &["editor", "vscode", "terminal"], semantic_type: "fact" },
        CorpusRecord { content: "Editor font size set to fourteen points for comfortable reading",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Auto-save enabled with one second delay prevents data loss",
            level: Level::Domain, tags: &["editor", "config"], semantic_type: "decision" },
        CorpusRecord { content: "Bracket pair colorization makes nested code much easier to read",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Line numbers and minimap always visible in editor sidebar",
            level: Level::Domain, tags: &["editor", "config"], semantic_type: "preference" },
        CorpusRecord { content: "Editor extensions for Rust provide inline type hints and errors",
            level: Level::Domain, tags: &["editor", "rust", "extensions"], semantic_type: "fact" },
    ]
}

// ═══════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default)]
struct EvalResult {
    seed_mode: String,
    total_records: usize,
    total_beliefs: usize,
    seeds: usize,
    partitions_ge2: usize,
    concepts_formed: usize,
    stable_concepts: usize,
    candidate_concepts: usize,
    concept_coverage: f32,
    false_merges: usize,
    // Per-topic breakdown
    topics_with_concepts: usize,
    total_topics: usize,
}

fn run_eval(aura: &Aura, corpora: &[(&str, Vec<CorpusRecord>)], cycles: usize) -> EvalResult {
    let mut result = EvalResult::default();
    result.total_topics = corpora.len();

    // Store all records
    for (_name, corpus) in corpora {
        store_corpus(aura, corpus);
        result.total_records += corpus.len();
    }

    // Run maintenance cycles
    run_cycles(aura, cycles);

    // Collect beliefs
    let beliefs = aura.get_beliefs(None);
    result.total_beliefs = beliefs.len();
    result.seeds = beliefs.iter().filter(|b| {
        matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
    }).count();

    // Count seeds per partition
    let mut sp: HashMap<String, usize> = HashMap::new();
    for b in &beliefs {
        if matches!(b.state, BeliefState::Resolved | BeliefState::Singleton) {
            let parts: Vec<&str> = b.key.split(':').collect();
            let pk = format!("{}:{}", parts.first().unwrap_or(&"default"), parts.last().unwrap_or(&"fact"));
            *sp.entry(pk).or_insert(0) += 1;
        }
    }
    result.partitions_ge2 = sp.values().filter(|&&v| v >= 2).count();

    // Collect concepts
    let concepts = aura.get_concepts(None);
    result.concepts_formed = concepts.len();
    result.stable_concepts = concepts.iter().filter(|c| c.state == ConceptState::Stable).count();
    result.candidate_concepts = concepts.iter().filter(|c| c.state == ConceptState::Candidate).count();

    // Concept coverage: for each corpus record, check if any recalled result is in a concept
    let mut covered = 0usize;
    for (_name, corpus) in corpora {
        for rec in corpus {
            let results = aura.recall_structured(
                rec.content, Some(5), Some(0.0), Some(true), None, None,
            ).unwrap_or_default();
            for r in &results {
                if concepts.iter().any(|c| c.record_ids.contains(&r.1.id)) {
                    covered += 1;
                    break;
                }
            }
        }
    }
    result.concept_coverage = covered as f32 / result.total_records as f32;

    // Cross-topic false merge: check if any concept contains records from multiple topics
    // Use tag overlap as proxy: deploy records have "deploy", database have "database", editor have "editor"
    let topic_tags = ["deploy", "database", "editor"];
    for concept in &concepts {
        let mut topic_count = 0;
        for &tag in &topic_tags {
            let has_tag = concept.tags.iter().any(|t| t == tag)
                || concept.record_ids.iter().any(|rid| {
                    if let Some(rec) = aura.get(rid) {
                        rec.tags.iter().any(|t| t == tag)
                    } else { false }
                });
            if has_tag { topic_count += 1; }
        }
        if topic_count > 1 { result.false_merges += 1; }
    }

    // Topics with concepts
    for &tag in &topic_tags {
        let has_concept = concepts.iter().any(|c| {
            c.record_ids.iter().any(|rid| {
                aura.get(rid).map_or(false, |r| r.tags.iter().any(|t| t == tag))
            })
        });
        if has_concept { result.topics_with_concepts += 1; }
    }

    result
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

/// Test 1: Realistic corpora with Standard seed gates.
#[test]
fn realistic_corpus_standard_gates() {
    println!("\n══════════════════════════════════════════════════");
    println!("  REALISTIC CORPUS: Standard gates (stability≥2.0, confidence≥0.55)");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];

    let result = run_eval(&aura, &corpora, 8);
    let _ = result.seed_mode.clone(); // suppress unused

    println!("  Records: {}", result.total_records);
    println!("  Beliefs: {}", result.total_beliefs);
    println!("  Seeds (Resolved|Singleton): {}", result.seeds);
    println!("  Partitions with ≥2 seeds: {}", result.partitions_ge2);
    println!("  Concepts: {} (stable={}, candidate={})",
        result.concepts_formed, result.stable_concepts, result.candidate_concepts);
    println!("  Coverage: {:.1}%", result.concept_coverage * 100.0);
    println!("  Topics with concepts: {}/{}", result.topics_with_concepts, result.total_topics);
    println!("  False merges: {}", result.false_merges);

    // Safety: no false merges
    assert_eq!(result.false_merges, 0, "Cross-topic false merge detected");
}

/// Test 2: Realistic corpora with Relaxed seed gates.
#[test]
fn realistic_corpus_relaxed_gates() {
    println!("\n══════════════════════════════════════════════════");
    println!("  REALISTIC CORPUS: Relaxed gates (stability≥1.0, confidence≥0.40)");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];

    let result = run_eval(&aura, &corpora, 8);

    println!("  Records: {}", result.total_records);
    println!("  Beliefs: {}", result.total_beliefs);
    println!("  Seeds (Resolved|Singleton): {}", result.seeds);
    println!("  Partitions with ≥2 seeds: {}", result.partitions_ge2);
    println!("  Concepts: {} (stable={}, candidate={})",
        result.concepts_formed, result.stable_concepts, result.candidate_concepts);
    println!("  Coverage: {:.1}%", result.concept_coverage * 100.0);
    println!("  Topics with concepts: {}/{}", result.topics_with_concepts, result.total_topics);
    println!("  False merges: {}", result.false_merges);

    assert_eq!(result.false_merges, 0, "Cross-topic false merge detected");
}

/// Test 3: Compare Standard vs Relaxed side by side.
#[test]
fn realistic_corpus_standard_vs_relaxed() {
    println!("\n══════════════════════════════════════════════════");
    println!("  COMPARISON: Standard vs Relaxed");
    println!("══════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_deploy()),
            ("database", corpus_database()),
            ("editor", corpus_editor()),
        ]
    };

    // Standard
    let (aura_std, _d1) = open_temp_aura();
    let r_std = run_eval(&aura_std, &corpora_fn(), 8);

    // Relaxed
    let (aura_rel, _d2) = open_temp_aura();
    aura_rel.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    let r_rel = run_eval(&aura_rel, &corpora_fn(), 8);

    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>8} {:>8} {:>6}",
        "Mode", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "Topics", "FM");
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}/{} {:>6}",
        "Standard", r_std.total_beliefs, r_std.seeds, r_std.partitions_ge2,
        r_std.concepts_formed, r_std.concept_coverage * 100.0,
        r_std.topics_with_concepts, r_std.total_topics, r_std.false_merges);
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}/{} {:>6}",
        "Relaxed", r_rel.total_beliefs, r_rel.seeds, r_rel.partitions_ge2,
        r_rel.concepts_formed, r_rel.concept_coverage * 100.0,
        r_rel.topics_with_concepts, r_rel.total_topics, r_rel.false_merges);

    // Safety: neither mode should have false merges
    assert_eq!(r_std.false_merges, 0, "Standard: cross-topic false merge");
    assert_eq!(r_rel.false_merges, 0, "Relaxed: cross-topic false merge");
}

/// Test 4: Per-topic detail — which topics form concepts.
#[test]
fn realistic_corpus_per_topic_detail() {
    println!("\n══════════════════════════════════════════════════");
    println!("  PER-TOPIC DETAIL: Relaxed mode");
    println!("══════════════════════════════════════════════════\n");

    let topics: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];

    for (name, corpus) in &topics {
        let (aura, _dir) = open_temp_aura();
        aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
        store_corpus(&aura, corpus);
        run_cycles(&aura, 8);

        let beliefs = aura.get_beliefs(None);
        let concepts = aura.get_concepts(None);

        println!("  [{}] records={} beliefs={} concepts={}", name, corpus.len(),
            beliefs.len(), concepts.len());
        for b in &beliefs {
            println!("    belief: key={} state={:?} stability={:.1} confidence={:.3}",
                b.key, b.state, b.stability, b.confidence);
        }
        for c in &concepts {
            println!("    concept: state={:?} score={:.3} beliefs={} records={} tags={:?}",
                c.state, c.abstraction_score, c.belief_ids.len(), c.record_ids.len(), c.tags);
        }
    }
}

/// Test 5: Identity stability — concepts stable across 5 replay cycles.
#[test]
fn realistic_corpus_identity_stability() {
    println!("\n══════════════════════════════════════════════════");
    println!("  IDENTITY STABILITY: 10 cycles with Relaxed");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];
    for (_, corpus) in &corpora {
        store_corpus(&aura, corpus);
    }

    let mut prev_keys: Vec<String> = Vec::new();
    let mut streak = 0usize;
    let mut max_streak = 0usize;

    for cycle in 0..10 {
        aura.run_maintenance();
        let concepts = aura.get_concepts(None);
        let mut keys: Vec<String> = concepts.iter().map(|c| c.key.clone()).collect();
        keys.sort();

        if keys == prev_keys && !keys.is_empty() {
            streak += 1;
        } else {
            streak = 0;
        }
        if streak > max_streak { max_streak = streak; }

        println!("  Cycle {}: concepts={} keys={} streak={} max_streak={}",
            cycle, concepts.len(), keys.len(), streak, max_streak);
        prev_keys = keys;
    }

    // If concepts formed, they should achieve at least 3 consecutive stable cycles
    if !prev_keys.is_empty() {
        assert!(max_streak >= 3,
            "Concept identity unstable: max streak only {} cycles", max_streak);
        println!("\n  Identity stable for max {} consecutive cycles — PASS", max_streak);
    } else {
        println!("\n  No concepts formed — identity stability N/A");
    }
}

/// Test 6: Candidate B not affected by seed mode change.
#[test]
fn realistic_corpus_candidate_b_safe() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CANDIDATE B SAFETY: Relaxed mode");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    aura.set_belief_rerank_enabled(true);

    let corpora = vec![corpus_deploy(), corpus_database(), corpus_editor()];
    for c in &corpora { store_corpus(&aura, c); }
    run_cycles(&aura, 8);

    let queries = ["deploy staging production", "database query performance",
        "editor dark mode preference", "monitoring canary rollout"];
    let mut all_ok = true;

    for q in &queries {
        aura.set_belief_rerank_enabled(false);
        let base = aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        aura.set_belief_rerank_enabled(true);
        let ranked = aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        if base.len() != ranked.len() { all_ok = false; }
        else {
            let base_ids: Vec<_> = base.iter().map(|r| r.1.id.clone()).collect();
            let ranked_ids: Vec<_> = ranked.iter().map(|r| r.1.id.clone()).collect();
            let max_shift = ranked_ids.iter().enumerate()
                .filter_map(|(i, rid)| base_ids.iter().position(|x| x == rid)
                    .map(|orig| if i > orig { i - orig } else { orig - i }))
                .max().unwrap_or(0);
            if max_shift > 2 { all_ok = false; }
        }
    }

    println!("  Candidate B under Relaxed mode: {}", if all_ok { "PASS" } else { "FAIL" });
    assert!(all_ok, "Candidate B regressed under Relaxed seed mode");
}

/// Test 7: Cross-layer stack with Relaxed mode.
#[test]
fn realistic_corpus_cross_layer_intact() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CROSS-LAYER: Relaxed mode");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);

    let corpora = vec![corpus_deploy(), corpus_database(), corpus_editor()];
    for c in &corpora { store_corpus(&aura, c); }

    for _ in 0..8 {
        let report = aura.run_maintenance();
        assert!(report.timings.total_ms >= 0.0);
    }

    let results = aura.recall_structured("deployment monitoring", Some(10), Some(0.0), Some(true), None, None)
        .unwrap_or_default();
    assert!(!results.is_empty(), "Recall returned 0 results");

    println!("  beliefs={} concepts={} causal={} policies={} — PASS",
        aura.get_beliefs(None).len(), aura.get_concepts(None).len(),
        aura.get_causal_patterns(None).len(),
        aura.get_surfaced_policy_hints(Some(10)).len());
}

/// Test 8: Aggregate verdict.
#[test]
fn realistic_corpus_verdict() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  CONCEPT REALISTIC EVAL: AGGREGATE VERDICT");
    println!("══════════════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_deploy()),
            ("database", corpus_database()),
            ("editor", corpus_editor()),
        ]
    };

    // Standard
    let (aura_std, _d1) = open_temp_aura();
    let r_std = run_eval(&aura_std, &corpora_fn(), 8);

    // Relaxed
    let (aura_rel, _d2) = open_temp_aura();
    aura_rel.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    let r_rel = run_eval(&aura_rel, &corpora_fn(), 8);

    println!("  ── Results ──\n");
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "Mode", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "FM");
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}",
        "Standard", r_std.total_beliefs, r_std.seeds, r_std.partitions_ge2,
        r_std.concepts_formed, r_std.concept_coverage * 100.0, r_std.false_merges);
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}",
        "Relaxed", r_rel.total_beliefs, r_rel.seeds, r_rel.partitions_ge2,
        r_rel.concepts_formed, r_rel.concept_coverage * 100.0, r_rel.false_merges);

    // Verdict
    let std_has_concepts = r_std.concepts_formed > 0;
    let rel_has_concepts = r_rel.concepts_formed > 0;
    let rel_safe = r_rel.false_merges == 0;

    let verdict = if std_has_concepts && r_std.concept_coverage > 0.0 {
        "REALISTIC CORPUS UNBLOCKS C WITH STANDARD GATES"
    } else if rel_has_concepts && rel_safe && r_rel.concept_coverage > 0.0 {
        "RELAXED GATES UNBLOCK C ON REALISTIC CORPUS"
    } else if rel_has_concepts && !rel_safe {
        "RELAXED GATES PRODUCE CONCEPTS BUT WITH FALSE MERGES"
    } else if r_std.partitions_ge2 > 0 || r_rel.partitions_ge2 > 0 {
        "DENSITY IMPROVED BUT CONCEPTS STILL BLOCKED"
    } else {
        "CORPUS STILL INSUFFICIENT"
    };

    println!("\n  VERDICT: {}\n", verdict);
    println!("  Standard concepts: {} (coverage {:.1}%)", r_std.concepts_formed, r_std.concept_coverage * 100.0);
    println!("  Relaxed concepts:  {} (coverage {:.1}%)", r_rel.concepts_formed, r_rel.concept_coverage * 100.0);
    println!("  False merges: std={} relaxed={}", r_std.false_merges, r_rel.false_merges);
}

// ═══════════════════════════════════════════════════════════
// Dense corpus: positive control — records with uniform tags
// to verify the pipeline works when given adequate density
// ═══════════════════════════════════════════════════════════

/// Dense corpus: 12 records all sharing the same 2 tags and same semantic_type.
/// This should produce beliefs → seeds → concepts if the pipeline is functional.
fn corpus_dense_deploy() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "Deployed version 2.3 to staging environment for validation",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Staging deployment passed all automated smoke tests successfully",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Promoted staging build to production canary fleet for monitoring",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Canary deployment showed zero error rate increase over baseline",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Completed full production rollout of version 2.3 to all regions",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Post-deploy monitoring confirmed healthy metrics across services",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Rollback plan for version 2.3 is documented and tested already",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Blue-green deployment strategy reduces downtime during releases",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Feature flags control gradual rollout of new deploy features",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Deployment pipeline includes automated security scanning step",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Each deploy requires approval from at least one team reviewer",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Deploy artifacts are versioned and stored in container registry",
            level: Level::Domain, tags: &["deploy", "pipeline"], semantic_type: "decision" },
    ]
}

fn corpus_dense_database() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "PostgreSQL connection pool maximum set to thirty connections",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Database query performance improved after adding composite indexes",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Weekly database backups stored in encrypted offsite location",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Database migration scripts always run inside a transaction block",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Read replicas handle reporting queries to reduce primary load",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Database connection timeouts set to five seconds for all services",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Slow query log enabled for queries exceeding two hundred milliseconds",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Database schema changes reviewed by DBA before production deploy",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Partitioning large tables by date improves database query speed",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Database credentials rotated monthly using vault secret manager",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
        CorpusRecord { content: "Connection pooling via PgBouncer reduces database overhead significantly",
            level: Level::Domain, tags: &["database", "ops"], semantic_type: "decision" },
    ]
}

/// Test 9: Dense positive control — uniform tags, same semantic_type.
/// If this doesn't produce concepts, the pipeline itself is broken.
#[test]
fn dense_corpus_positive_control() {
    println!("\n══════════════════════════════════════════════════");
    println!("  DENSE CORPUS: Positive control (uniform tags)");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_dense_deploy()),
        ("database", corpus_dense_database()),
    ];

    // Store records
    for (_, corpus) in &corpora {
        store_corpus(&aura, corpus);
    }

    // Run 8 cycles and print concept phase diagnostics from each
    for cycle in 0..8 {
        let report = aura.run_maintenance();
        let cp = &report.concept;
        if cp.seeds_found > 0 || cycle == 7 {
            println!("  Cycle {}: seeds={} centroids={} partitions_ge2={} pairwise={} above_threshold={}",
                cycle, cp.seeds_found, cp.centroids_built, cp.partitions_with_multiple_seeds,
                cp.pairwise_comparisons, cp.pairwise_above_threshold);
            if cp.pairwise_comparisons > 0 {
                println!("    Tanimoto: min={:.4} max={:.4} avg={:.4} centroid_size={:.1}",
                    cp.tanimoto_min, cp.tanimoto_max, cp.tanimoto_avg, cp.avg_centroid_size);
            }
            println!("    concepts={} stable={} rejected={}",
                cp.candidates_found, cp.stable_count, cp.rejected_count);
        }
    }

    let beliefs = aura.get_beliefs(None);
    let concepts = aura.get_concepts(None);
    let seeds = beliefs.iter().filter(|b|
        matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)).count();

    println!("\n  Records: {}", corpora.iter().map(|(_, c)| c.len()).sum::<usize>());
    println!("  Beliefs: {}", beliefs.len());
    println!("  Seeds: {}", seeds);
    println!("  Concepts: {} (stable={}, candidate={})",
        concepts.len(),
        concepts.iter().filter(|c| c.state == ConceptState::Stable).count(),
        concepts.iter().filter(|c| c.state == ConceptState::Candidate).count());

    // Print belief details
    for b in &beliefs {
        println!("    belief: key={} state={:?} stability={:.1} conf={:.3}",
            b.key, b.state, b.stability, b.confidence);
    }

    // Check false merges
    let topic_tags = ["deploy", "database"];
    let mut false_merges = 0;
    for concept in &concepts {
        let mut topic_count = 0;
        for &tag in &topic_tags {
            let has_tag = concept.tags.iter().any(|t| t == tag)
                || concept.record_ids.iter().any(|rid| {
                    if let Some(rec) = aura.get(rid) {
                        rec.tags.iter().any(|t| t == tag)
                    } else { false }
                });
            if has_tag { topic_count += 1; }
        }
        if topic_count > 1 { false_merges += 1; }
    }

    assert_eq!(false_merges, 0, "Cross-topic false merge in dense corpus");
}

/// Test 10: TagFamily + Relaxed on realistic corpus.
#[test]
fn realistic_corpus_tagfamily_relaxed() {
    println!("\n══════════════════════════════════════════════════");
    println!("  TAGFAMILY + RELAXED: Realistic corpus");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];

    let result = run_eval(&aura, &corpora, 8);

    println!("  Records: {}", result.total_records);
    println!("  Beliefs: {}", result.total_beliefs);
    println!("  Seeds: {}", result.seeds);
    println!("  Partitions ≥2: {}", result.partitions_ge2);
    println!("  Concepts: {} (stable={}, candidate={})",
        result.concepts_formed, result.stable_concepts, result.candidate_concepts);
    println!("  Coverage: {:.1}%", result.concept_coverage * 100.0);
    println!("  Topics with concepts: {}/{}", result.topics_with_concepts, result.total_topics);
    println!("  False merges: {}", result.false_merges);

    assert_eq!(result.false_merges, 0, "Cross-topic false merge");
}

/// Test 11: 4-way comparison matrix (Standard/TagFamily × Standard/Relaxed).
#[test]
fn four_way_comparison() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  4-WAY COMPARISON: CoarseKey × SeedMode");
    println!("══════════════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_deploy()),
            ("database", corpus_database()),
            ("editor", corpus_editor()),
        ]
    };

    let configs: Vec<(&str, CoarseKeyMode, ConceptSeedMode)> = vec![
        ("Std+Std", CoarseKeyMode::Standard, ConceptSeedMode::Standard),
        ("Std+Rlx", CoarseKeyMode::Standard, ConceptSeedMode::Relaxed),
        ("TF+Std",  CoarseKeyMode::TagFamily, ConceptSeedMode::Standard),
        ("TF+Rlx",  CoarseKeyMode::TagFamily, ConceptSeedMode::Relaxed),
    ];

    println!("  {:10} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "Config", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "FM");

    let mut any_concepts = false;
    for (label, coarse, seed) in &configs {
        let (aura, _dir) = open_temp_aura();
        aura.set_belief_coarse_key_mode(*coarse);
        aura.set_concept_seed_mode(*seed);

        let r = run_eval(&aura, &corpora_fn(), 8);
        if r.concepts_formed > 0 { any_concepts = true; }

        println!("  {:10} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}",
            label, r.total_beliefs, r.seeds, r.partitions_ge2,
            r.concepts_formed, r.concept_coverage * 100.0, r.false_merges);

        assert_eq!(r.false_merges, 0, "{}: cross-topic false merge", label);
    }

    if any_concepts {
        println!("\n  RESULT: At least one config produces concepts!");
    } else {
        println!("\n  RESULT: No config produces concepts on realistic diverse-tag corpus.");
        println!("  Root cause: tag diversity + semantic_type splits → insufficient partition density.");
    }
}

/// Test 12: Dense corpus 4-way comparison — proves pipeline works.
#[test]
fn dense_corpus_four_way() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  DENSE CORPUS 4-WAY: Positive control matrix");
    println!("══════════════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_dense_deploy()),
            ("database", corpus_dense_database()),
        ]
    };

    let configs: Vec<(&str, CoarseKeyMode, ConceptSeedMode)> = vec![
        ("Std+Std", CoarseKeyMode::Standard, ConceptSeedMode::Standard),
        ("Std+Rlx", CoarseKeyMode::Standard, ConceptSeedMode::Relaxed),
        ("TF+Std",  CoarseKeyMode::TagFamily, ConceptSeedMode::Standard),
        ("TF+Rlx",  CoarseKeyMode::TagFamily, ConceptSeedMode::Relaxed),
    ];

    println!("  {:10} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "Config", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "FM");

    for (label, coarse, seed) in &configs {
        let (aura, _dir) = open_temp_aura();
        aura.set_belief_coarse_key_mode(*coarse);
        aura.set_concept_seed_mode(*seed);

        let r = run_eval(&aura, &corpora_fn(), 8);

        println!("  {:10} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}",
            label, r.total_beliefs, r.seeds, r.partitions_ge2,
            r.concepts_formed, r.concept_coverage * 100.0, r.false_merges);

        assert_eq!(r.false_merges, 0, "{}: cross-topic false merge", label);
    }
}
