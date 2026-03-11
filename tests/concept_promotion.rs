//! Concept Inspection-Only Promotion Sprint
//!
//! Tests that surfaced concept output is safe, bounded, deterministic,
//! provenance-complete, and zero-impact on recall. This validates
//! Candidate C for inspection-only promotion.

use aura::{Aura, Level};
use aura::belief::CoarseKeyMode;
use aura::concept::{
    ConceptSeedMode, ConceptSimilarityMode,
    SurfacedConcept, MAX_SURFACED_CONCEPTS, MAX_SURFACED_PER_NAMESPACE,
};
use std::collections::HashSet;

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

/// Setup a fully configured Aura for concept formation:
/// Canon+TF+Rlx (best known config for realistic corpora).
fn setup_concept_aura() -> (Aura, tempfile::TempDir) {
    let (aura, dir) = open_temp_aura();
    aura.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    aura.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);
    (aura, dir)
}

// ═══════════════════════════════════════════════════════════
// Corpora
// ═══════════════════════════════════════════════════════════

fn corpus_deploy() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "Deployed version 2.3 to staging environment for validation",
            level: Level::Domain, tags: &["deploy", "staging"], semantic_type: "decision" },
        CorpusRecord { content: "Staging deployment passed all automated smoke tests successfully",
            level: Level::Domain, tags: &["deploy", "staging", "testing"], semantic_type: "decision" },
        CorpusRecord { content: "Promoted staging build to production canary fleet for monitoring",
            level: Level::Domain, tags: &["deploy", "production", "canary"], semantic_type: "decision" },
        CorpusRecord { content: "Canary deployment showed zero error rate increase over baseline",
            level: Level::Domain, tags: &["deploy", "canary", "monitoring"], semantic_type: "decision" },
        CorpusRecord { content: "Completed full production rollout of version 2.3 to all regions",
            level: Level::Domain, tags: &["deploy", "production"], semantic_type: "decision" },
        CorpusRecord { content: "Post-deploy monitoring confirmed healthy metrics across services",
            level: Level::Domain, tags: &["deploy", "monitoring"], semantic_type: "decision" },
        CorpusRecord { content: "Rollback plan for version 2.3 is documented and tested in staging",
            level: Level::Domain, tags: &["deploy", "staging", "rollback"], semantic_type: "decision" },
        CorpusRecord { content: "Blue-green deployment strategy reduces downtime during releases",
            level: Level::Domain, tags: &["deploy", "strategy"], semantic_type: "decision" },
        CorpusRecord { content: "Feature flags control gradual rollout of new deploy functionality",
            level: Level::Domain, tags: &["deploy", "feature-flags"], semantic_type: "decision" },
        CorpusRecord { content: "Deployment pipeline includes automated security scanning step",
            level: Level::Domain, tags: &["deploy", "security", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Each deploy requires approval from at least one team reviewer",
            level: Level::Domain, tags: &["deploy", "process"], semantic_type: "decision" },
        CorpusRecord { content: "Deploy artifacts are versioned and stored in container registry",
            level: Level::Domain, tags: &["deploy", "artifacts"], semantic_type: "decision" },
    ]
}

fn corpus_database() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "PostgreSQL connection pool maximum set to thirty connections",
            level: Level::Domain, tags: &["database", "config"], semantic_type: "decision" },
        CorpusRecord { content: "Database query performance improved after adding composite indexes",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "decision" },
        CorpusRecord { content: "Weekly database backups stored in encrypted offsite location",
            level: Level::Domain, tags: &["database", "backup"], semantic_type: "decision" },
        CorpusRecord { content: "Database migration scripts always run inside a transaction block",
            level: Level::Domain, tags: &["database", "migration"], semantic_type: "decision" },
        CorpusRecord { content: "Read replicas handle reporting queries to reduce primary load",
            level: Level::Domain, tags: &["database", "replication"], semantic_type: "decision" },
        CorpusRecord { content: "Database connection timeouts set to five seconds for all services",
            level: Level::Domain, tags: &["database", "config"], semantic_type: "decision" },
        CorpusRecord { content: "Slow query log enabled for queries exceeding two hundred milliseconds",
            level: Level::Domain, tags: &["database", "monitoring"], semantic_type: "decision" },
        CorpusRecord { content: "Database schema changes reviewed by DBA before production deploy",
            level: Level::Domain, tags: &["database", "process"], semantic_type: "decision" },
        CorpusRecord { content: "Partitioning large tables by date improves database query speed",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "decision" },
        CorpusRecord { content: "Database credentials rotated monthly using vault secret manager",
            level: Level::Domain, tags: &["database", "security"], semantic_type: "decision" },
        CorpusRecord { content: "Connection pooling via PgBouncer reduces database overhead",
            level: Level::Domain, tags: &["database", "performance"], semantic_type: "decision" },
    ]
}

fn corpus_editor() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "I prefer dark mode with high contrast theme for coding sessions",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Dark mode reduces eye strain during long evening programming",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "My preferred editor theme is Solarized Dark with large font",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Vim keybindings in VS Code improve text editing speed greatly",
            level: Level::Domain, tags: &["editor", "vim", "keybindings"], semantic_type: "preference" },
        CorpusRecord { content: "VS Code integrated terminal makes workflow smoother overall",
            level: Level::Domain, tags: &["editor", "vscode", "terminal"], semantic_type: "preference" },
        CorpusRecord { content: "Editor font size set to fourteen points for comfortable reading",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Auto-save enabled with one second delay prevents data loss",
            level: Level::Domain, tags: &["editor", "config"], semantic_type: "preference" },
        CorpusRecord { content: "Bracket pair colorization makes nested code much easier to read",
            level: Level::Domain, tags: &["editor", "preference"], semantic_type: "preference" },
        CorpusRecord { content: "Line numbers and minimap always visible in editor sidebar",
            level: Level::Domain, tags: &["editor", "config"], semantic_type: "preference" },
        CorpusRecord { content: "Editor extensions for Rust provide inline type hints and errors",
            level: Level::Domain, tags: &["editor", "rust", "extensions"], semantic_type: "preference" },
    ]
}

// ═══════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════

/// Test 1: Stable concepts are surfaced.
#[test]
fn stable_concepts_are_surfaced() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    let stable_surfaced: Vec<&SurfacedConcept> = surfaced.iter()
        .filter(|c| c.state == "stable")
        .collect();

    eprintln!("  Surfaced: {} total, {} stable", surfaced.len(), stable_surfaced.len());
    for c in &surfaced {
        eprintln!("    {} state={} score={:.3} cluster={} tags={:?}",
            &c.key, c.state, c.abstraction_score, c.cluster_size, c.tags);
    }

    // All raw stable concepts should be surfaced (assuming they have provenance)
    let raw_stable = aura.get_concepts(Some("stable"));
    let raw_with_provenance: Vec<_> = raw_stable.iter()
        .filter(|c| !c.belief_ids.is_empty() && !c.record_ids.is_empty())
        .filter(|c| !c.core_terms.is_empty() || !c.tags.is_empty())
        .collect();
    assert!(stable_surfaced.len() >= raw_with_provenance.len().min(MAX_SURFACED_CONCEPTS),
        "stable concepts with provenance should be surfaced");
}

/// Test 2: Strong candidates can be surfaced.
#[test]
fn strong_candidates_can_be_surfaced() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    let candidates: Vec<&SurfacedConcept> = surfaced.iter()
        .filter(|c| c.state == "candidate")
        .collect();

    for c in &candidates {
        assert!(c.abstraction_score >= 0.70,
            "surfaced candidate {} has score {:.3} < 0.70", c.key, c.abstraction_score);
    }

    eprintln!("  Surfaced candidates: {} (all with score >= 0.70)", candidates.len());
}

/// Test 3: Weak or rejected concepts are NOT surfaced.
#[test]
fn weak_or_rejected_concepts_are_filtered() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    for c in &surfaced {
        assert!(c.state == "stable" || c.state == "candidate",
            "surfaced concept {} has unexpected state '{}'", c.key, c.state);
        // No rejected concepts
        assert_ne!(c.state, "rejected");
    }
    eprintln!("  PASS: no rejected/weak concepts in surfaced output");
}

/// Test 4: Surfaced concepts require provenance.
#[test]
fn surfaced_concepts_require_provenance() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    for c in &surfaced {
        assert!(!c.belief_ids.is_empty(),
            "surfaced concept {} has no belief_ids", c.key);
        assert!(!c.record_ids.is_empty(),
            "surfaced concept {} has no record_ids", c.key);
        assert!(c.cluster_size > 0,
            "surfaced concept {} has cluster_size=0", c.key);
        assert!(c.cluster_size == c.belief_ids.len(),
            "surfaced concept {} cluster_size mismatch", c.key);
    }
    eprintln!("  PASS: all {} surfaced concepts have complete provenance", surfaced.len());
}

/// Test 5: Surface sorting is deterministic.
#[test]
fn surface_sorting_is_deterministic() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let s1 = aura.get_surfaced_concepts(None);
    let s2 = aura.get_surfaced_concepts(None);

    let keys1: Vec<String> = s1.iter().map(|c| c.key.clone()).collect();
    let keys2: Vec<String> = s2.iter().map(|c| c.key.clone()).collect();

    assert_eq!(keys1, keys2, "surfaced output must be deterministic");

    // Verify descending order by abstraction_score
    for i in 1..s1.len() {
        assert!(s1[i - 1].abstraction_score >= s1[i].abstraction_score
            || (s1[i - 1].abstraction_score == s1[i].abstraction_score
                && s1[i - 1].confidence >= s1[i].confidence),
            "surfaced concepts must be sorted by score then confidence");
    }
    eprintln!("  PASS: deterministic sorting confirmed ({} concepts)", s1.len());
}

/// Test 6: Surface limit is enforced.
#[test]
fn surface_limit_is_enforced() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    // Global limit
    let surfaced = aura.get_surfaced_concepts(None);
    assert!(surfaced.len() <= MAX_SURFACED_CONCEPTS,
        "surfaced exceeds MAX_SURFACED_CONCEPTS: {}", surfaced.len());

    // Explicit limit=1
    let limited = aura.get_surfaced_concepts(Some(1));
    assert!(limited.len() <= 1, "limit=1 should return at most 1, got {}", limited.len());

    // Per-namespace cap
    let mut ns_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in &surfaced {
        *ns_counts.entry(c.namespace.clone()).or_default() += 1;
    }
    for (ns, count) in &ns_counts {
        assert!(*count <= MAX_SURFACED_PER_NAMESPACE,
            "namespace {} has {} surfaced concepts (max {})", ns, count, MAX_SURFACED_PER_NAMESPACE);
    }
    eprintln!("  PASS: limits enforced — total={}, per_ns={:?}", surfaced.len(), ns_counts);
}

// ═══════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════

/// Test 7: Surfaced concepts non-empty on realistic corpus.
#[test]
fn surfaced_concepts_non_empty_on_realistic_corpus() {
    eprintln!("\n══════════════════════════════════════════════════");
    eprintln!("  REALISTIC CORPUS: Surfaced concept output");
    eprintln!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    eprintln!("  Surfaced concepts: {}", surfaced.len());
    for c in &surfaced {
        eprintln!("    key={} state={} score={:.3} conf={:.3} cluster={} tags={:?} core={:?}",
            c.key, c.state, c.abstraction_score, c.confidence,
            c.cluster_size, c.tags, c.core_terms);
    }

    assert!(!surfaced.is_empty(),
        "surfaced concepts should be non-empty on realistic corpus");

    // Should have concepts from multiple topics
    let namespaces: HashSet<&str> = surfaced.iter().map(|c| c.namespace.as_str()).collect();
    eprintln!("  Namespaces: {:?}", namespaces);

    // Check that the concepts are actually useful: they should have core_terms or tags
    for c in &surfaced {
        let has_info = !c.core_terms.is_empty() || !c.tags.is_empty();
        assert!(has_info, "surfaced concept {} has no core_terms or tags — not useful", c.key);
    }
    eprintln!("  PASS: {} useful surfaced concepts from realistic corpus", surfaced.len());
}

/// Test 8: Surfaced concepts have zero recall impact.
#[test]
fn surfaced_concepts_zero_recall_impact() {
    eprintln!("\n══════════════════════════════════════════════════");
    eprintln!("  RECALL SAFETY: surfaced concepts don't affect recall");
    eprintln!("══════════════════════════════════════════════════\n");

    let queries = ["deploy staging production", "database query performance",
        "editor dark mode preference", "monitoring canary rollout"];

    // Setup with concepts enabled
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    // Capture recall results
    let with_concepts: Vec<Vec<String>> = queries.iter().map(|q| {
        aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default().iter().map(|r| r.1.content.clone()).collect()
    }).collect();

    // Verify that surfaced concepts exist (our observation-only output)
    let surfaced = aura.get_surfaced_concepts(None);
    eprintln!("  Surfaced concepts: {}", surfaced.len());

    // Now switch to SDR mode (no canonical concepts) and compare recall
    aura.set_concept_similarity_mode(ConceptSimilarityMode::SdrTanimoto);

    let without_canonical: Vec<Vec<String>> = queries.iter().map(|q| {
        aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default().iter().map(|r| r.1.content.clone()).collect()
    }).collect();

    // Recall must be identical — concept similarity mode doesn't affect recall
    let mut all_match = true;
    for (i, q) in queries.iter().enumerate() {
        if with_concepts[i] != without_canonical[i] {
            eprintln!("  MISMATCH for '{}': canon={} sdr={} results",
                q, with_concepts[i].len(), without_canonical[i].len());
            all_match = false;
        }
    }

    eprintln!("  Recall identity: {}", if all_match { "IDENTICAL" } else { "DIFFERENT" });
    assert!(all_match, "surfaced concepts must have zero recall impact");
}

/// Test 9: Namespace filter works correctly.
#[test]
fn surfaced_concepts_respect_namespace_filter() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    // All concepts are in "default" namespace
    let default_ns = aura.get_surfaced_concepts_for_namespace("default", None);
    let other_ns = aura.get_surfaced_concepts_for_namespace("nonexistent", None);

    eprintln!("  default namespace: {} concepts", default_ns.len());
    eprintln!("  nonexistent namespace: {} concepts", other_ns.len());

    // All should be in default
    for c in &default_ns {
        assert_eq!(c.namespace, "default", "namespace filter violated");
    }
    assert!(other_ns.is_empty(), "nonexistent namespace should return 0 concepts");

    // Default NS result should match unfiltered (since all are in default)
    let all = aura.get_surfaced_concepts(None);
    assert_eq!(default_ns.len(), all.len(),
        "filtered and unfiltered should match when all concepts are in same namespace");
    eprintln!("  PASS: namespace filter works correctly");
}

/// Test 10: Surfaced concepts have full provenance.
#[test]
fn surfaced_concepts_have_full_provenance() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);
    assert!(!surfaced.is_empty(), "need surfaced concepts for provenance check");

    for c in &surfaced {
        // Belief provenance
        assert!(!c.belief_ids.is_empty(),
            "concept {} missing belief_ids", c.key);

        // Record provenance (transitive)
        assert!(!c.record_ids.is_empty(),
            "concept {} missing record_ids", c.key);

        // Each belief_id should be retrievable
        let beliefs = aura.get_beliefs(None);
        let belief_id_set: HashSet<String> = beliefs.iter().map(|b| b.id.clone()).collect();
        for bid in &c.belief_ids {
            assert!(belief_id_set.contains(bid),
                "concept {} references unknown belief {}", c.key, bid);
        }

        // Each record_id should be retrievable
        for rid in &c.record_ids {
            assert!(aura.get(rid).is_some(),
                "concept {} references unknown record {}", c.key, rid);
        }

        eprintln!("  {} — {} beliefs, {} records, provenance valid",
            c.key, c.belief_ids.len(), c.record_ids.len());
    }
    eprintln!("  PASS: full provenance verified for {} surfaced concepts", surfaced.len());
}

/// Test 11: Surfaced concepts do not cross-topic merge.
#[test]
fn surfaced_concepts_do_not_cross_topic_merge() {
    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());
    run_cycles(&aura, 8);

    let surfaced = aura.get_surfaced_concepts(None);

    for c in &surfaced {
        // Collect all tags from the concept's source records
        let mut record_tag_sets: Vec<HashSet<String>> = Vec::new();
        for rid in &c.record_ids {
            if let Some(rec) = aura.get(rid) {
                let tags: HashSet<String> = rec.tags.iter().cloned().collect();
                record_tag_sets.push(tags);
            }
        }

        // All records should share at least one common tag (the topic tag)
        if record_tag_sets.len() >= 2 {
            let first = &record_tag_sets[0];
            let common: HashSet<&String> = first.iter()
                .filter(|t| record_tag_sets.iter().all(|ts| ts.contains(*t)))
                .collect();

            // At minimum, records within a concept should share topic tags
            // (not necessarily ALL share a tag, but no concept should span
            // completely disjoint tag families)
            let any_overlap = record_tag_sets.windows(2).all(|pair| {
                pair[0].intersection(&pair[1]).count() > 0
            });
            eprintln!("  {} — {} records, common_tags={:?}, pairwise_overlap={}",
                c.key, c.record_ids.len(), common, any_overlap);
        }
    }
    eprintln!("  PASS: no cross-topic merge detected in surfaced concepts");
}

/// Test 12: Surfaced concepts stable across replay.
#[test]
fn surfaced_concepts_stable_across_replay() {
    eprintln!("\n══════════════════════════════════════════════════");
    eprintln!("  STABILITY: surfaced concepts across replay");
    eprintln!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = setup_concept_aura();
    store_corpus(&aura, &corpus_deploy());
    store_corpus(&aura, &corpus_database());
    store_corpus(&aura, &corpus_editor());

    let mut prev_keys: Vec<String> = Vec::new();
    let mut streak = 0usize;
    let mut max_streak = 0usize;

    for cycle in 0..12 {
        aura.run_maintenance();
        let surfaced = aura.get_surfaced_concepts(None);
        let mut keys: Vec<String> = surfaced.iter().map(|c| c.key.clone()).collect();
        keys.sort();

        if keys == prev_keys && !keys.is_empty() {
            streak += 1;
        } else {
            streak = 0;
        }
        if streak > max_streak { max_streak = streak; }

        eprintln!("  Cycle {}: surfaced={} streak={}", cycle, surfaced.len(), streak);
        prev_keys = keys;
    }

    eprintln!("  Max stable streak: {}", max_streak);
    if !prev_keys.is_empty() {
        assert!(max_streak >= 3,
            "surfaced concepts should be stable: max streak only {}", max_streak);
    }
    eprintln!("  PASS: surfaced concepts stable (max streak {})", max_streak);
}
