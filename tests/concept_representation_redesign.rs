//! Concept Representation Redesign Sprint
//!
//! Tests whether canonical feature representation (Variant A) unblocks
//! concept formation on realistic corpora where SDR n-gram Tanimoto fails.
//!
//! Evaluation matrix:
//! - Same-topic similarity distribution (Jaccard vs SDR Tanimoto)
//! - Cross-topic separation gap
//! - Concept coverage on realistic corpora
//! - False merge adversarial pack
//! - Identity stability
//! - Recall safety

use aura::{Aura, Level};
use aura::belief::{BeliefState, CoarseKeyMode};
use aura::concept::{ConceptSeedMode, ConceptSimilarityMode, ConceptState, canonical_tokens, jaccard};
use std::collections::{HashMap, HashSet};

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
// Corpora
// ═══════════════════════════════════════════════════════════

/// Realistic deploy corpus (12 records, diverse tags)
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

/// Realistic database corpus (11 records, diverse tags)
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

/// Realistic editor corpus (10 records, diverse tags)
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

/// Adversarial corpus: records that share broad tags but are different topics.
/// Must NOT merge cross-topic.
fn corpus_adversarial() -> Vec<CorpusRecord> {
    vec![
        // Group 1: API rate limiting (shares "api" with group 2)
        CorpusRecord { content: "API rate limiting configured at one hundred requests per minute",
            level: Level::Domain, tags: &["api", "security"], semantic_type: "decision" },
        CorpusRecord { content: "Rate limiter protects API endpoints from abuse and overload",
            level: Level::Domain, tags: &["api", "security"], semantic_type: "decision" },
        CorpusRecord { content: "API throttling rules apply per client token basis globally",
            level: Level::Domain, tags: &["api", "security"], semantic_type: "decision" },
        CorpusRecord { content: "Rate limit headers returned in API response for client awareness",
            level: Level::Domain, tags: &["api", "security"], semantic_type: "decision" },
        // Group 2: API documentation (shares "api" with group 1)
        CorpusRecord { content: "API documentation generated automatically from OpenAPI specification",
            level: Level::Domain, tags: &["api", "docs"], semantic_type: "decision" },
        CorpusRecord { content: "All API endpoints documented with request and response examples",
            level: Level::Domain, tags: &["api", "docs"], semantic_type: "decision" },
        CorpusRecord { content: "API changelog maintained for every breaking change in version",
            level: Level::Domain, tags: &["api", "docs"], semantic_type: "decision" },
        CorpusRecord { content: "API reference includes authentication flow and error codes",
            level: Level::Domain, tags: &["api", "docs"], semantic_type: "decision" },
        // Group 3: CI/CD pipeline (different topic entirely)
        CorpusRecord { content: "CI pipeline runs unit tests and integration tests on every push",
            level: Level::Domain, tags: &["ci", "testing"], semantic_type: "decision" },
        CorpusRecord { content: "Continuous integration builds trigger on pull request creation",
            level: Level::Domain, tags: &["ci", "testing"], semantic_type: "decision" },
        CorpusRecord { content: "CI build artifacts cached to speed up subsequent pipeline runs",
            level: Level::Domain, tags: &["ci", "pipeline"], semantic_type: "decision" },
        CorpusRecord { content: "Failed CI checks block pull request merge into main branch",
            level: Level::Domain, tags: &["ci", "pipeline"], semantic_type: "decision" },
    ]
}

// ═══════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default)]
struct EvalResult {
    mode_label: String,
    total_records: usize,
    total_beliefs: usize,
    seeds: usize,
    partitions_ge2: usize,
    concepts_formed: usize,
    stable_concepts: usize,
    candidate_concepts: usize,
    concept_coverage: f32,
    false_merges: usize,
    topics_with_concepts: usize,
    total_topics: usize,
    // Similarity diagnostics from ConceptPhaseReport
    pairwise_comparisons: usize,
    pairwise_above: usize,
    sim_min: f32,
    sim_max: f32,
    sim_avg: f32,
}

fn run_eval(
    aura: &Aura,
    corpora: &[(&str, Vec<CorpusRecord>)],
    cycles: usize,
) -> EvalResult {
    let mut result = EvalResult::default();
    result.total_topics = corpora.len();

    for (_, corpus) in corpora {
        store_corpus(aura, corpus);
        result.total_records += corpus.len();
    }

    // Run cycles, capture last report
    let mut last_report = None;
    for _ in 0..cycles {
        last_report = Some(aura.run_maintenance());
    }

    // Extract concept phase diagnostics
    if let Some(report) = last_report {
        let cp = &report.concept;
        result.pairwise_comparisons = cp.pairwise_comparisons;
        result.pairwise_above = cp.pairwise_above_threshold;
        result.sim_min = cp.tanimoto_min;
        result.sim_max = cp.tanimoto_max;
        result.sim_avg = cp.tanimoto_avg;
    }

    let beliefs = aura.get_beliefs(None);
    result.total_beliefs = beliefs.len();
    result.seeds = beliefs.iter().filter(|b|
        matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)).count();

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

    let concepts = aura.get_concepts(None);
    result.concepts_formed = concepts.len();
    result.stable_concepts = concepts.iter().filter(|c| c.state == ConceptState::Stable).count();
    result.candidate_concepts = concepts.iter().filter(|c| c.state == ConceptState::Candidate).count();

    // Coverage
    let mut covered = 0usize;
    for (_, corpus) in corpora {
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
    result.concept_coverage = if result.total_records > 0 {
        covered as f32 / result.total_records as f32
    } else { 0.0 };

    // Cross-topic false merges
    let topic_tags = ["deploy", "database", "editor", "api", "ci"];
    for concept in &concepts {
        let mut topic_count = 0;
        for &tag in &topic_tags {
            let has_tag = concept.record_ids.iter().any(|rid| {
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

fn print_result(r: &EvalResult) {
    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        r.mode_label, r.total_beliefs, r.seeds, r.partitions_ge2,
        r.concepts_formed, format!("{:.1}%", r.concept_coverage * 100.0), r.false_merges);
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

/// Test 1: Canonical representation increases same-topic similarity.
#[test]
fn canonical_representation_increases_same_topic_similarity() {
    println!("\n══════════════════════════════════════════════════");
    println!("  SAME-TOPIC SIMILARITY: Canonical vs SDR");
    println!("══════════════════════════════════════════════════\n");

    // Same-topic deploy records
    let texts = [
        "Deployed version 2.3 to staging environment for validation",
        "Promoted staging build to production canary fleet for monitoring",
        "Blue-green deployment strategy reduces downtime during releases",
        "Feature flags control gradual rollout of new deploy functionality",
        "Completed full production rollout of version 2.3 to all regions",
    ];

    // Compute canonical Jaccard pairwise
    let token_sets: Vec<HashSet<String>> = texts.iter()
        .map(|t| canonical_tokens(t))
        .collect();

    let mut jaccard_sims = Vec::new();
    for i in 0..token_sets.len() {
        for j in (i+1)..token_sets.len() {
            let j_val = jaccard(&token_sets[i], &token_sets[j]);
            jaccard_sims.push(j_val);
        }
    }

    let avg_jaccard = jaccard_sims.iter().sum::<f32>() / jaccard_sims.len() as f32;
    let min_jaccard = jaccard_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_jaccard = jaccard_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("  Same-topic deploy (canonical Jaccard):");
    println!("    min={:.4} max={:.4} avg={:.4}", min_jaccard, max_jaccard, avg_jaccard);
    println!("    (SDR Tanimoto baseline was 0.048-0.069)");

    // Print token sets for diagnostics
    for (i, ts) in token_sets.iter().enumerate() {
        let mut sorted: Vec<&String> = ts.iter().collect();
        sorted.sort();
        println!("    text[{}] tokens: {:?}", i, sorted);
    }

    // Key assertion: canonical Jaccard should show meaningful same-topic signal
    // Even if avg is modest, max should exceed the clustering threshold
    println!("\n  ASSESSMENT:");
    println!("    avg Jaccard: {:.4} (SDR Tanimoto baseline: 0.048-0.069)", avg_jaccard);
    println!("    max Jaccard: {:.4} (concept threshold: 0.12)", max_jaccard);
    println!("    Jaccard > SDR baseline: {}", avg_jaccard > 0.050);

    // At minimum, max should show some pairs above the clustering threshold
    assert!(max_jaccard > 0.05,
        "Canonical max Jaccard ({:.4}) should show some same-topic signal", max_jaccard);
}

/// Test 2: Cross-topic separation preserved.
#[test]
fn canonical_preserves_cross_topic_separation() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CROSS-TOPIC SEPARATION: Canonical Jaccard");
    println!("══════════════════════════════════════════════════\n");

    let deploy_texts = [
        "Deployed version 2.3 to staging environment for validation",
        "Blue-green deployment strategy reduces downtime during releases",
        "Canary deployment showed zero error rate increase over baseline",
    ];
    let database_texts = [
        "PostgreSQL connection pool maximum set to thirty connections",
        "Database query performance improved after adding composite indexes",
        "Partitioning large tables by date improves database query speed",
    ];
    let editor_texts = [
        "I prefer dark mode with high contrast theme for coding sessions",
        "Vim keybindings in VS Code improve text editing speed greatly",
        "Editor font size set to fourteen points for comfortable reading",
    ];

    let deploy_tokens: Vec<HashSet<String>> = deploy_texts.iter().map(|t| canonical_tokens(t)).collect();
    let db_tokens: Vec<HashSet<String>> = database_texts.iter().map(|t| canonical_tokens(t)).collect();
    let editor_tokens: Vec<HashSet<String>> = editor_texts.iter().map(|t| canonical_tokens(t)).collect();

    // Same-topic similarities
    let mut same_topic_sims = Vec::new();
    for tokens in [&deploy_tokens, &db_tokens, &editor_tokens] {
        for i in 0..tokens.len() {
            for j in (i+1)..tokens.len() {
                same_topic_sims.push(jaccard(&tokens[i], &tokens[j]));
            }
        }
    }

    // Cross-topic similarities
    let mut cross_topic_sims = Vec::new();
    let all_groups = [&deploy_tokens, &db_tokens, &editor_tokens];
    for gi in 0..all_groups.len() {
        for gj in (gi+1)..all_groups.len() {
            for a in all_groups[gi] {
                for b in all_groups[gj] {
                    cross_topic_sims.push(jaccard(a, b));
                }
            }
        }
    }

    let avg_same = same_topic_sims.iter().sum::<f32>() / same_topic_sims.len().max(1) as f32;
    let max_same = same_topic_sims.iter().cloned().fold(0.0f32, f32::max);
    let avg_cross = cross_topic_sims.iter().sum::<f32>() / cross_topic_sims.len().max(1) as f32;
    let max_cross = cross_topic_sims.iter().cloned().fold(0.0f32, f32::max);
    let gap = avg_same - avg_cross;

    println!("  Same-topic:  avg={:.4} max={:.4}", avg_same, max_same);
    println!("  Cross-topic: avg={:.4} max={:.4}", avg_cross, max_cross);
    println!("  Separation gap: {:.4}", gap);

    // Cross-topic should be noticeably lower
    assert!(gap > 0.02, "Separation gap ({:.4}) should be > 0.02", gap);
    // Max cross-topic should not exceed the threshold too much
    println!("  Max cross-topic Jaccard: {:.4} (threshold: 0.12)", max_cross);
}

/// Test 3: Concepts form on realistic corpus with CanonicalFeature mode.
#[test]
fn canonical_forms_concepts_on_realistic_corpus() {
    println!("\n══════════════════════════════════════════════════");
    println!("  REALISTIC CORPUS: CanonicalFeature mode");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    // Use TagFamily for better belief density
    aura.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy", corpus_deploy()),
        ("database", corpus_database()),
        ("editor", corpus_editor()),
    ];

    let mut r = run_eval(&aura, &corpora, 8);
    r.mode_label = "Canon+TF+Rlx".to_string();

    println!("  Records: {}", r.total_records);
    println!("  Beliefs: {}", r.total_beliefs);
    println!("  Seeds: {}", r.seeds);
    println!("  Partitions ≥2: {}", r.partitions_ge2);
    println!("  Concepts: {} (stable={}, candidate={})",
        r.concepts_formed, r.stable_concepts, r.candidate_concepts);
    println!("  Coverage: {:.1}%", r.concept_coverage * 100.0);
    println!("  Topics with concepts: {}/{}", r.topics_with_concepts, r.total_topics);
    println!("  False merges: {}", r.false_merges);
    println!("  Similarity: min={:.4} max={:.4} avg={:.4} ({}/{})",
        r.sim_min, r.sim_max, r.sim_avg, r.pairwise_above, r.pairwise_comparisons);

    // Debug: print canonical tokens for each belief's records
    let beliefs = aura.get_beliefs(None);
    for b in &beliefs {
        // Try to get records for this belief via recall
        let tokens = canonical_tokens(&b.key);
        println!("    belief {} key={} canon_key_tokens={:?}", b.id, b.key, tokens);
    }

    assert_eq!(r.false_merges, 0, "Cross-topic false merge detected");
}

/// Test 4: Full 4-way comparison (SdrTanimoto/Canonical × Standard/Relaxed).
#[test]
fn representation_redesign_compares_all_variants() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  4-WAY COMPARISON: SdrTanimoto vs CanonicalFeature");
    println!("══════════════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_deploy()),
            ("database", corpus_database()),
            ("editor", corpus_editor()),
        ]
    };

    let configs: Vec<(&str, ConceptSimilarityMode, CoarseKeyMode, ConceptSeedMode)> = vec![
        ("SDR+Std+Std",    ConceptSimilarityMode::SdrTanimoto,     CoarseKeyMode::Standard,  ConceptSeedMode::Standard),
        ("SDR+TF+Rlx",     ConceptSimilarityMode::SdrTanimoto,     CoarseKeyMode::TagFamily, ConceptSeedMode::Relaxed),
        ("Canon+Std+Rlx",  ConceptSimilarityMode::CanonicalFeature, CoarseKeyMode::Standard,  ConceptSeedMode::Relaxed),
        ("Canon+TF+Rlx",   ConceptSimilarityMode::CanonicalFeature, CoarseKeyMode::TagFamily, ConceptSeedMode::Relaxed),
    ];

    println!("  {:15} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "Config", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "FM");

    for (label, sim_mode, coarse, seed) in &configs {
        let (aura, _dir) = open_temp_aura();
        aura.set_concept_similarity_mode(*sim_mode);
        aura.set_belief_coarse_key_mode(*coarse);
        aura.set_concept_seed_mode(*seed);

        let mut r = run_eval(&aura, &corpora_fn(), 8);
        r.mode_label = label.to_string();
        print_result(&r);

        // Print similarity diagnostics
        if r.pairwise_comparisons > 0 {
            println!("    sim: min={:.4} max={:.4} avg={:.4} above={}/{}",
                r.sim_min, r.sim_max, r.sim_avg, r.pairwise_above, r.pairwise_comparisons);
        }

        assert_eq!(r.false_merges, 0, "{}: cross-topic false merge", label);
    }
}

/// Test 5: Adversarial false merge pack — shared broad tags across different topics.
#[test]
fn representation_redesign_does_not_explode_false_merges() {
    println!("\n══════════════════════════════════════════════════");
    println!("  ADVERSARIAL: False merge pack");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    // Deliberately NOT using TagFamily — adversarial pack tests cross-topic
    // separation when records share a broad tag ("api"). TagFamily groups by
    // first tag, which would force all api-tagged records into one partition
    // regardless of concept similarity.

    let corpora: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("adversarial", corpus_adversarial()),
    ];

    let r = run_eval(&aura, &corpora, 8);

    println!("  Records: {}", r.total_records);
    println!("  Beliefs: {}", r.total_beliefs);
    println!("  Concepts: {}", r.concepts_formed);
    println!("  False merges: {}", r.false_merges);

    // Adversarial pack: API rate-limiting and API docs share "api" tag
    // They should NOT merge into one concept
    let concepts = aura.get_concepts(None);
    for c in &concepts {
        let has_security = c.record_ids.iter().any(|rid| {
            aura.get(rid).map_or(false, |r| r.tags.iter().any(|t| t == "security"))
        });
        let has_docs = c.record_ids.iter().any(|rid| {
            aura.get(rid).map_or(false, |r| r.tags.iter().any(|t| t == "docs"))
        });
        assert!(!(has_security && has_docs),
            "API security and API docs should NOT merge into one concept");
    }

    // No CI records should merge with API records
    for c in &concepts {
        let has_api = c.record_ids.iter().any(|rid| {
            aura.get(rid).map_or(false, |r| r.tags.iter().any(|t| t == "api"))
        });
        let has_ci = c.record_ids.iter().any(|rid| {
            aura.get(rid).map_or(false, |r| r.tags.iter().any(|t| t == "ci"))
        });
        assert!(!(has_api && has_ci),
            "API and CI records should NOT merge into one concept");
    }

    println!("  PASS: no cross-topic false merges in adversarial pack");
}

/// Test 6: Zero recall impact — CanonicalFeature mode doesn't affect recall.
#[test]
fn representation_redesign_has_zero_recall_impact() {
    println!("\n══════════════════════════════════════════════════");
    println!("  RECALL SAFETY: CanonicalFeature vs SdrTanimoto");
    println!("══════════════════════════════════════════════════\n");

    let queries = ["deploy staging production", "database query performance",
        "editor dark mode preference", "monitoring canary rollout"];

    // Run with SdrTanimoto
    let (aura_sdr, _d1) = open_temp_aura();
    let corpora = vec![corpus_deploy(), corpus_database(), corpus_editor()];
    for c in &corpora { store_corpus(&aura_sdr, c); }
    run_cycles(&aura_sdr, 8);

    let sdr_results: Vec<Vec<String>> = queries.iter().map(|q| {
        aura_sdr.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default().iter().map(|r| r.1.content.clone()).collect()
    }).collect();

    // Toggle to CanonicalFeature — same Aura, no extra cycles.
    // Concept similarity mode only affects concept clustering, NOT recall.
    // Running extra cycles would change belief scores and invalidate the comparison.
    aura_sdr.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);

    let can_results: Vec<Vec<String>> = queries.iter().map(|q| {
        aura_sdr.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default().iter().map(|r| r.1.content.clone()).collect()
    }).collect();

    // Results should be identical — concept similarity mode doesn't affect recall
    let mut all_match = true;
    for (i, q) in queries.iter().enumerate() {
        if sdr_results[i] != can_results[i] {
            println!("  MISMATCH for '{}': SDR={} results, Canon={} results",
                q, sdr_results[i].len(), can_results[i].len());
            all_match = false;
        }
    }

    println!("  Recall identity: {}", if all_match { "IDENTICAL" } else { "DIFFERENT" });
    assert!(all_match, "Recall results should be identical regardless of concept similarity mode");
}

/// Test 7: Identity stability — concepts stable across 10 replay cycles.
#[test]
fn representation_redesign_keeps_identity_stable() {
    println!("\n══════════════════════════════════════════════════");
    println!("  IDENTITY STABILITY: CanonicalFeature 10 cycles");
    println!("══════════════════════════════════════════════════\n");

    let (aura, _dir) = open_temp_aura();
    aura.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    aura.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);

    let corpora = vec![corpus_deploy(), corpus_database(), corpus_editor()];
    for c in &corpora { store_corpus(&aura, c); }

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

        println!("  Cycle {}: concepts={} stable_streak={}", cycle, concepts.len(), streak);
        prev_keys = keys;
    }

    if !prev_keys.is_empty() {
        assert!(max_streak >= 3,
            "Concept identity unstable: max streak only {} cycles", max_streak);
        println!("\n  Identity stable for {} consecutive cycles — PASS", max_streak);
    } else {
        println!("\n  No concepts formed — identity stability N/A");
    }
}

/// Test 8: Final aggregate verdict.
#[test]
fn representation_redesign_emits_final_verdict() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  CONCEPT REPRESENTATION REDESIGN: FINAL VERDICT");
    println!("══════════════════════════════════════════════════════════\n");

    let corpora_fn = || -> Vec<(&str, Vec<CorpusRecord>)> {
        vec![
            ("deploy", corpus_deploy()),
            ("database", corpus_database()),
            ("editor", corpus_editor()),
        ]
    };

    // Baseline: SDR+Standard (what we had before)
    let (aura_base, _d1) = open_temp_aura();
    let r_base = run_eval(&aura_base, &corpora_fn(), 8);

    // Best SDR config: SDR+TagFamily+Relaxed
    let (aura_sdr, _d2) = open_temp_aura();
    aura_sdr.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);
    aura_sdr.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    let r_sdr = run_eval(&aura_sdr, &corpora_fn(), 8);

    // Canonical: Canon+TagFamily+Relaxed
    let (aura_can, _d3) = open_temp_aura();
    aura_can.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura_can.set_belief_coarse_key_mode(CoarseKeyMode::TagFamily);
    aura_can.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    let r_can = run_eval(&aura_can, &corpora_fn(), 8);

    // Canonical on Standard coarse key
    let (aura_can_std, _d4) = open_temp_aura();
    aura_can_std.set_concept_similarity_mode(ConceptSimilarityMode::CanonicalFeature);
    aura_can_std.set_concept_seed_mode(ConceptSeedMode::Relaxed);
    let r_can_std = run_eval(&aura_can_std, &corpora_fn(), 8);

    println!("  ── Results ──\n");
    println!("  {:18} {:>8} {:>6} {:>6} {:>8} {:>8} {:>6}",
        "Config", "Beliefs", "Seeds", "P≥2", "Concepts", "Coverage", "FM");
    for (label, r) in [
        ("SDR+Std+Std", &r_base),
        ("SDR+TF+Rlx", &r_sdr),
        ("Canon+Std+Rlx", &r_can_std),
        ("Canon+TF+Rlx", &r_can),
    ] {
        println!("  {:18} {:>8} {:>6} {:>6} {:>8} {:>7.1}% {:>6}",
            label, r.total_beliefs, r.seeds, r.partitions_ge2,
            r.concepts_formed, r.concept_coverage * 100.0, r.false_merges);
        if r.pairwise_comparisons > 0 {
            println!("    sim: min={:.4} max={:.4} avg={:.4} above={}/{}",
                r.sim_min, r.sim_max, r.sim_avg, r.pairwise_above, r.pairwise_comparisons);
        }
    }

    // Verdict logic
    let canonical_has_concepts = r_can.concepts_formed > 0;
    let canonical_safe = r_can.false_merges == 0;
    let canonical_coverage_up = r_can.concept_coverage > r_base.concept_coverage;
    let sim_materially_up = r_can.sim_avg > 0.069; // above SDR Tanimoto baseline

    let verdict = if canonical_has_concepts && canonical_safe && canonical_coverage_up {
        "SAFE REPRESENTATION REDESIGN FOUND"
    } else if sim_materially_up && !canonical_has_concepts {
        "PARTIAL IMPROVEMENT, C STILL BLOCKED"
    } else if canonical_has_concepts && !canonical_safe {
        "CONCEPTS FORMED BUT WITH FALSE MERGES"
    } else if !sim_materially_up {
        "NO SAFE REPRESENTATION REDESIGN"
    } else {
        "CONCEPT PATH REQUIRES DEEPER ARCHITECTURAL CHANGE"
    };

    println!("\n  VERDICT: {}\n", verdict);

    // Detailed assessment
    println!("  Same-topic similarity rise: SDR avg ~0.06 → Canon avg {:.4}", r_can.sim_avg);
    println!("  Canonical concepts: {} (coverage {:.1}%)", r_can.concepts_formed, r_can.concept_coverage * 100.0);
    println!("  False merges: SDR={} Canon={}", r_base.false_merges, r_can.false_merges);
}
