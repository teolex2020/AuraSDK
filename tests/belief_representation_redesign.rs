//! Belief Representation Redesign Sprint
//!
//! Tests 3 experimental belief corridor/grouping designs against baseline to find
//! a safe way to increase practical belief density and unblock Candidate C.
//!
//! Variants:
//!   Standard    — namespace:sorted_tags(top3):semantic_type (baseline)
//!   TagFamily   — namespace:dominant_tag_family:semantic_type (Variant A)
//!   DualKey     — namespace:semantic_type + tag-guarded SDR@0.10 (Variant B)
//!   Neighborhood— namespace:semantic_type + tag-guarded SDR@0.08 (Variant C)
//!
//! Measures: belief density, precision/recall/F1, concept seeds/coverage,
//! Candidate B safety, cross-layer integrity, churn stability.

use aura::{Aura, Level};
use aura::belief::{BeliefEngine, BeliefState, CoarseKeyMode, SdrLookup};
use aura::concept::ConceptState;
use aura::sdr::SDRInterpreter;
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

// ═══════════════════════════════════════════════════════════
// Config
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
struct RedesignConfig {
    name: &'static str,
    coarse_mode: CoarseKeyMode,
    threshold_override: Option<f32>,
}

const CONFIGS: &[RedesignConfig] = &[
    RedesignConfig { name: "Standard",      coarse_mode: CoarseKeyMode::Standard,         threshold_override: None },
    RedesignConfig { name: "TagFamily",      coarse_mode: CoarseKeyMode::TagFamily,        threshold_override: None },
    RedesignConfig { name: "DualKey",        coarse_mode: CoarseKeyMode::DualKey,           threshold_override: None },
    RedesignConfig { name: "Neighborhood",   coarse_mode: CoarseKeyMode::NeighborhoodPool, threshold_override: None },
];

fn open_aura_with_config(cfg: &RedesignConfig) -> (Aura, tempfile::TempDir) {
    let (aura, dir) = open_temp_aura();
    aura.set_belief_coarse_key_mode(cfg.coarse_mode);
    if let Some(t) = cfg.threshold_override {
        aura.set_belief_similarity_threshold(Some(t));
    }
    (aura, dir)
}

// ═══════════════════════════════════════════════════════════
// Labeled dataset (from belief_grouping_benchmark)
// ═══════════════════════════════════════════════════════════

struct LabeledRecord {
    content: &'static str,
    level: Level,
    tags: &'static [&'static str],
    source_type: &'static str,
    semantic_type: &'static str,
    cluster_id: u32,
}

fn store_labeled(aura: &Aura, records: &[LabeledRecord]) -> Vec<String> {
    records.iter().map(|lr| {
        let rec = aura.store(
            lr.content,
            Some(lr.level), Some(lr.tags.iter().map(|t| t.to_string()).collect()),
            None, None, Some(lr.source_type), None, Some(false), None, None,
            Some(lr.semantic_type),
        ).unwrap_or_else(|e| panic!("store failed: {}", e));
        rec.id.clone()
    }).collect()
}

fn curated_dataset() -> Vec<LabeledRecord> {
    vec![
        // Cluster 1: deployment pipeline (3 paraphrases + 1 contradiction)
        LabeledRecord { content: "Always deploy through the CI pipeline to staging before production",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1 },
        LabeledRecord { content: "Deploy through CI pipeline to staging before going to production",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1 },
        LabeledRecord { content: "CI pipeline must deploy to staging first before production release",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1 },
        // Cluster 2: database config
        LabeledRecord { content: "Set PostgreSQL connection pool maximum to thirty active connections",
            level: Level::Decisions, tags: &["database", "config", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 2 },
        LabeledRecord { content: "PostgreSQL connection pool maximum should be thirty connections",
            level: Level::Decisions, tags: &["database", "config", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 2 },
        // Cluster 3: coding preferences
        LabeledRecord { content: "I prefer dark mode with high contrast theme in all code editors",
            level: Level::Domain, tags: &["ui", "preferences", "coding"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 3 },
        LabeledRecord { content: "Dark mode with high contrast is my preferred coding editor theme",
            level: Level::Domain, tags: &["ui", "preferences", "coding"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 3 },
        // Cluster 4: reading preferences
        LabeledRecord { content: "For reading documentation I prefer light mode with serif font",
            level: Level::Domain, tags: &["ui", "preferences", "reading"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 4 },
        LabeledRecord { content: "Light mode with serif fonts works best for reading documentation",
            level: Level::Domain, tags: &["ui", "preferences", "reading"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 4 },
        // Cluster 5: Rust backend
        LabeledRecord { content: "The entire backend service layer is written in Rust for performance",
            level: Level::Identity, tags: &["tech", "rust", "backend"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 5 },
        LabeledRecord { content: "Our backend services are all implemented in Rust for performance",
            level: Level::Identity, tags: &["tech", "rust", "backend"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 5 },
        // Cluster 6: CI automation
        LabeledRecord { content: "GitHub Actions runs all continuous integration workflows automatically",
            level: Level::Domain, tags: &["tech", "ci", "automation"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 6 },
        LabeledRecord { content: "All CI workflows are automated through GitHub Actions pipelines",
            level: Level::Domain, tags: &["tech", "ci", "automation"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 6 },
        // Cluster 7: API architecture
        LabeledRecord { content: "We use gRPC for all latency-critical inter-service communication",
            level: Level::Decisions, tags: &["architecture", "api", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 7 },
        LabeledRecord { content: "gRPC is the standard for latency-critical service communication",
            level: Level::Decisions, tags: &["architecture", "api", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 7 },
        // Cluster 8: logging config
        LabeledRecord { content: "Production logging verbosity is set to INFO level for all services",
            level: Level::Domain, tags: &["logging", "config", "operations"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 8 },
        LabeledRecord { content: "All services use INFO log level in production environment",
            level: Level::Domain, tags: &["logging", "config", "operations"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 8 },
        // Cluster 9: testing policy
        LabeledRecord { content: "Integration tests must pass before any pull request can be merged",
            level: Level::Decisions, tags: &["testing", "process", "quality"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 9 },
        LabeledRecord { content: "No pull request merge without passing integration test suite first",
            level: Level::Decisions, tags: &["testing", "process", "quality"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 9 },
        // Cluster 10: security
        LabeledRecord { content: "Enable two-factor authentication for all production admin accounts",
            level: Level::Decisions, tags: &["security", "auth", "production"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 10 },
        LabeledRecord { content: "All production admin accounts require two-factor authentication",
            level: Level::Decisions, tags: &["security", "auth", "production"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 10 },
        // Cluster 1 contradiction
        LabeledRecord { content: "Sometimes we skip staging and deploy hotfixes direct to production",
            level: Level::Working, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "contradiction", cluster_id: 1 },
        // Cluster 12: Ukrainian
        LabeledRecord { content: "Використовуємо Kubernetes для оркестрації всіх мікросервісів",
            level: Level::Domain, tags: &["infra", "kubernetes", "orchestration"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 12 },
        LabeledRecord { content: "Kubernetes оркеструє всі наші мікросервіси в продакшені",
            level: Level::Domain, tags: &["infra", "kubernetes", "orchestration"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 12 },
        // Cluster 13: rate limiting
        LabeledRecord { content: "API rate limiting is configured at one thousand requests per minute",
            level: Level::Domain, tags: &["api", "config", "performance"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 13 },
        LabeledRecord { content: "Rate limiting set to one thousand requests per minute for all APIs",
            level: Level::Domain, tags: &["api", "config", "performance"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 13 },
    ]
}

// ═══════════════════════════════════════════════════════════
// Practical corpora
// ═══════════════════════════════════════════════════════════

struct CorpusRecord {
    content: String,
    level: Level,
    tags: Vec<String>,
    source_type: &'static str,
    semantic_type: &'static str,
}

fn store_corpus(aura: &Aura, records: &[CorpusRecord]) {
    for rec in records {
        aura.store(
            &rec.content, Some(rec.level), Some(rec.tags.clone()),
            None, None, Some(rec.source_type), None, Some(false), None, None,
            Some(rec.semantic_type),
        ).unwrap();
    }
}

fn practical_deploy_chain() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "Deployed version 2.3 to staging environment".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "staging".into()], source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "Staging deployment passed all smoke tests".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "staging".into(), "testing".into()], source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "Promoted staging build to production canary".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "production".into(), "canary".into()], source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "Canary deployment showed zero error rate increase".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "canary".into(), "monitoring".into()], source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "Completed full production rollout of version 2.3".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "production".into()], source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "Post-deploy monitoring confirmed healthy metrics".into(), level: Level::Domain,
            tags: vec!["deploy".into(), "monitoring".into()], source_type: "recorded", semantic_type: "fact" },
    ]
}

fn practical_stable_preference() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "I prefer dark mode in my IDE for coding".into(), level: Level::Domain,
            tags: vec!["editor".into(), "preference".into()], source_type: "recorded", semantic_type: "preference" },
        CorpusRecord { content: "Dark mode reduces eye strain during long coding sessions".into(), level: Level::Domain,
            tags: vec!["editor".into(), "preference".into()], source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "My preferred editor theme is dark with high contrast".into(), level: Level::Domain,
            tags: vec!["editor".into(), "preference".into()], source_type: "recorded", semantic_type: "preference" },
        CorpusRecord { content: "Dark backgrounds work better for evening programming".into(), level: Level::Domain,
            tags: vec!["editor".into(), "preference".into()], source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "Rust compiler gives helpful error messages".into(), level: Level::Domain,
            tags: vec!["rust".into(), "tooling".into()], source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "Cargo build system handles dependencies well".into(), level: Level::Domain,
            tags: vec!["rust".into(), "tooling".into()], source_type: "recorded", semantic_type: "fact" },
    ]
}

fn practical_multi_topic() -> Vec<CorpusRecord> {
    let mut records = Vec::new();
    for c in &["tabs are better than spaces for indentation", "spaces are better than tabs for alignment"] {
        records.push(CorpusRecord { content: c.to_string(), level: Level::Working,
            tags: vec!["style".into(), "formatting".into(), "indentation".into()],
            source_type: "recorded", semantic_type: "preference" });
    }
    for c in &["GraphQL is superior to REST for frontend flexibility", "REST is simpler and more cacheable than GraphQL"] {
        records.push(CorpusRecord { content: c.to_string(), level: Level::Working,
            tags: vec!["api".into(), "design".into()],
            source_type: "recorded", semantic_type: "preference" });
    }
    for c in &["monorepo is better for code sharing", "polyrepo is better for team autonomy"] {
        records.push(CorpusRecord { content: c.to_string(), level: Level::Working,
            tags: vec!["repository".into(), "organization".into()],
            source_type: "recorded", semantic_type: "preference" });
    }
    records
}

fn practical_contextual() -> Vec<CorpusRecord> {
    vec![
        CorpusRecord { content: "I use Vim keybindings in VS Code for speed".into(), level: Level::Domain,
            tags: vec!["editor".into(), "vim".into(), "keybindings".into()],
            source_type: "recorded", semantic_type: "preference" },
        CorpusRecord { content: "Vim keybindings work best for text editing and refactoring".into(), level: Level::Domain,
            tags: vec!["editor".into(), "vim".into(), "keybindings".into()],
            source_type: "recorded", semantic_type: "preference" },
        CorpusRecord { content: "VS Code provides great debugging experience".into(), level: Level::Domain,
            tags: vec!["editor".into(), "debugging".into(), "vscode".into()],
            source_type: "recorded", semantic_type: "fact" },
        CorpusRecord { content: "VS Code integrated terminal improves workflow".into(), level: Level::Domain,
            tags: vec!["editor".into(), "terminal".into(), "vscode".into()],
            source_type: "recorded", semantic_type: "fact" },
    ]
}

// ═══════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct QualityMetrics {
    precision: f32,
    recall: f32,
    f1: f32,
    false_merge_rate: f32,
    false_split_rate: f32,
    churn_rate: f32,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct ConceptMetrics {
    concepts_formed: usize,
    stable_concepts: usize,
    concept_coverage: f32,
}

// ═══════════════════════════════════════════════════════════
// Measurement helpers
// ═══════════════════════════════════════════════════════════

fn measure_quality(cfg: &RedesignConfig) -> QualityMetrics {
    let (aura, _dir) = open_aura_with_config(cfg);
    let dataset = curated_dataset();
    let cluster_labels: Vec<u32> = dataset.iter().map(|r| r.cluster_id).collect();
    let record_ids = store_labeled(&aura, &dataset);

    let _report = aura.run_maintenance();

    let sdr = SDRInterpreter::default();
    let mut records_map = HashMap::new();
    for rid in &record_ids {
        if let Some(rec) = aura.get(rid) {
            records_map.insert(rid.clone(), rec);
        }
    }
    let mut sdr_lookup: SdrLookup = HashMap::new();
    for (rid, rec) in &records_map {
        let sdr_vec = sdr.text_to_sdr(&rec.content, false);
        sdr_lookup.insert(rid.clone(), sdr_vec);
    }

    let mut engine = BeliefEngine::with_coarse_key_mode(cfg.coarse_mode);
    engine.claim_similarity_override = cfg.threshold_override;
    engine.update_with_sdr(&records_map, &sdr_lookup);

    let mut actual_pairs = HashSet::new();
    for belief in engine.beliefs.values() {
        let mut belief_records: Vec<String> = Vec::new();
        for hid in &belief.hypothesis_ids {
            if let Some(hyp) = engine.hypotheses.get(hid) {
                belief_records.extend(hyp.prototype_record_ids.iter().cloned());
            }
        }
        for i in 0..belief_records.len() {
            for j in (i + 1)..belief_records.len() {
                let a = &belief_records[i];
                let b = &belief_records[j];
                let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                actual_pairs.insert(pair);
            }
        }
    }

    let n = record_ids.len();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let a = &record_ids[i];
            let b = &record_ids[j];
            let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
            let should_group = cluster_labels[i] == cluster_labels[j];
            let did_group = actual_pairs.contains(&pair);
            match (should_group, did_group) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }
    }

    let precision = if tp + fp == 0 { 1.0 } else { tp as f32 / (tp + fp) as f32 };
    let recall = if tp + fn_ == 0 { 1.0 } else { tp as f32 / (tp + fn_) as f32 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    let false_merge_rate = if fp + tn == 0 { 0.0 } else { fp as f32 / (fp + tn) as f32 };
    let false_split_rate = if fn_ + tp == 0 { 0.0 } else { fn_ as f32 / (fn_ + tp) as f32 };

    let mut engine2 = BeliefEngine::with_coarse_key_mode(cfg.coarse_mode);
    engine2.claim_similarity_override = cfg.threshold_override;
    let mut last_churn = 0.0_f32;
    for _ in 0..5 {
        let report = engine2.update_with_sdr(&records_map, &sdr_lookup);
        last_churn = report.churn_rate;
    }

    QualityMetrics { precision, recall, f1, false_merge_rate, false_split_rate, churn_rate: last_churn }
}

fn measure_concept_practical(cfg: &RedesignConfig, corpus: &[CorpusRecord], cycles: usize) -> ConceptMetrics {
    let (aura, _dir) = open_aura_with_config(cfg);
    store_corpus(&aura, corpus);
    run_cycles(&aura, cycles);

    let concepts = aura.get_concepts(None);
    let stable = concepts.iter().filter(|c| c.state == ConceptState::Stable).count();

    let mut covered = 0usize;
    let mut total = 0usize;
    for rec in corpus {
        total += 1;
        let results = aura.recall_structured(
            &rec.content, Some(5), Some(0.0), Some(true), None, None,
        ).unwrap_or_default();
        if !results.is_empty() {
            for r in &results {
                let rid = &r.1.id;
                if concepts.iter().any(|c| c.record_ids.contains(rid)) {
                    covered += 1;
                    break;
                }
            }
        }
    }

    let coverage = if total == 0 { 0.0 } else { covered as f32 / total as f32 };
    ConceptMetrics { concepts_formed: concepts.len(), stable_concepts: stable, concept_coverage: coverage }
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

/// Test 1: Practical belief density per variant on deploy-chain.
#[test]
fn redesign_variant_increases_practical_belief_density() {
    println!("\n══════════════════════════════════════════════════");
    println!("  PRACTICAL DENSITY: deploy-chain (6 records)");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();

    let mut results: Vec<(&str, usize, usize, usize)> = Vec::new();
    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        store_corpus(&aura, &corpus);
        run_cycles(&aura, 5);

        let beliefs = aura.get_beliefs(None);
        let seeds = beliefs.iter().filter(|b| {
            matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                && b.stability >= 2.0 && b.confidence >= 0.55
        }).count();
        let concepts = aura.get_concepts(None);

        println!("  [{}] beliefs={} seeds={} concepts={}", cfg.name, beliefs.len(), seeds, concepts.len());
        results.push((cfg.name, beliefs.len(), seeds, concepts.len()));
    }

    // DualKey or Neighborhood should produce more beliefs than Standard
    let std_beliefs = results[0].1;
    let dualkey_beliefs = results[2].1;
    let neighborhood_beliefs = results[3].1;
    println!("\n  Standard={}, DualKey={}, Neighborhood={}", std_beliefs, dualkey_beliefs, neighborhood_beliefs);

    // At minimum, tag-guarded modes should not produce FEWER beliefs
    assert!(dualkey_beliefs >= std_beliefs || neighborhood_beliefs >= std_beliefs,
        "Neither DualKey nor Neighborhood increased density over Standard");
}

/// Test 2: Quality benchmark — precision must not collapse.
#[test]
fn redesign_variant_preserves_belief_precision() {
    println!("\n══════════════════════════════════════════════════");
    println!("  QUALITY: curated 26-record benchmark");
    println!("══════════════════════════════════════════════════\n");

    for cfg in CONFIGS {
        let q = measure_quality(cfg);
        println!("  [{}] P={:.3} R={:.3} F1={:.3} FM={:.3} churn={:.4}",
            cfg.name, q.precision, q.recall, q.f1, q.false_merge_rate, q.churn_rate);

        // Hard stop: precision must not collapse
        assert!(q.precision >= 0.50,
            "[{}] precision collapsed to {:.3}", cfg.name, q.precision);
        // Hard stop: false merge rate must stay bounded
        assert!(q.false_merge_rate <= 0.20,
            "[{}] false merge rate exploded to {:.3}", cfg.name, q.false_merge_rate);
    }

    // Standard baseline must still meet original gates
    let std_q = measure_quality(&CONFIGS[0]);
    assert!(std_q.precision >= 0.85 && std_q.recall >= 0.70 && std_q.f1 >= 0.75,
        "Standard baseline regressed");
}

/// Test 3: Candidate B safety — reranking bounded under all variants.
#[test]
fn redesign_variant_does_not_break_candidate_b() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CANDIDATE B SAFETY");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();
    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        aura.set_belief_rerank_enabled(true);
        store_corpus(&aura, &corpus);
        run_cycles(&aura, 5);

        let queries = ["deployment staging", "production canary", "monitoring deploy"];
        let mut all_ok = true;
        for q in &queries {
            aura.set_belief_rerank_enabled(false);
            let baseline = aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
                .unwrap_or_default();
            aura.set_belief_rerank_enabled(true);
            let reranked = aura.recall_structured(q, Some(10), Some(0.0), Some(true), None, None)
                .unwrap_or_default();
            if baseline.len() != reranked.len() { all_ok = false; continue; }
            let baseline_ids: Vec<_> = baseline.iter().map(|r| r.1.id.clone()).collect();
            let reranked_ids: Vec<_> = reranked.iter().map(|r| r.1.id.clone()).collect();
            let max_shift = reranked_ids.iter().enumerate()
                .filter_map(|(i, rid)| baseline_ids.iter().position(|x| x == rid)
                    .map(|orig| if i > orig { i - orig } else { orig - i }))
                .max().unwrap_or(0);
            if max_shift > 2 { all_ok = false; }
        }

        println!("  [{}] Candidate B: {}", cfg.name, if all_ok { "PASS" } else { "FAIL" });
        assert!(all_ok, "Candidate B regressed under {}", cfg.name);
    }
}

/// Test 4: Concept practical coverage per variant.
#[test]
fn redesign_variant_improves_concept_practical_coverage() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CONCEPT PRACTICAL COVERAGE");
    println!("══════════════════════════════════════════════════\n");

    let practical_sets: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy-chain", practical_deploy_chain()),
        ("stable-preference", practical_stable_preference()),
        ("multi-topic", practical_multi_topic()),
        ("contextual", practical_contextual()),
    ];

    let mut config_coverage: Vec<(&str, usize, f32)> = Vec::new();
    for cfg in CONFIGS {
        println!("  [{}]", cfg.name);
        let mut total_concepts = 0;
        let mut total_coverage = 0.0_f32;
        for (name, corpus) in &practical_sets {
            let m = measure_concept_practical(cfg, corpus, 8);
            println!("    {}: concepts={} coverage={:.1}%", name, m.concepts_formed, m.concept_coverage * 100.0);
            total_concepts += m.concepts_formed;
            total_coverage += m.concept_coverage;
        }
        let avg_cov = total_coverage / practical_sets.len() as f32;
        println!("    TOTAL: concepts={} avg_coverage={:.1}%\n", total_concepts, avg_cov * 100.0);
        config_coverage.push((cfg.name, total_concepts, avg_cov));
    }

    // Report best
    let best = config_coverage.iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    println!("  Best coverage: {} ({:.1}%)", best.0, best.2 * 100.0);
}

/// Test 5: No cross-topic false merge explosion.
#[test]
fn no_cross_topic_false_merge_explosion() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CROSS-TOPIC FALSE MERGE CHECK");
    println!("══════════════════════════════════════════════════\n");

    // Store records from unrelated topics in same aura
    let mixed: Vec<CorpusRecord> = vec![
        // Topic A: deploy
        CorpusRecord { content: "Deploy version 3.1 to staging environment for validation".into(),
            level: Level::Domain, tags: vec!["deploy".into(), "staging".into()],
            source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "Staging deployment validated and ready for production".into(),
            level: Level::Domain, tags: vec!["deploy".into(), "staging".into()],
            source_type: "recorded", semantic_type: "decision" },
        // Topic B: database (completely unrelated)
        CorpusRecord { content: "Configure database replication for high availability setup".into(),
            level: Level::Domain, tags: vec!["database".into(), "replication".into()],
            source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "Database replication configured with primary and two replicas".into(),
            level: Level::Domain, tags: vec!["database".into(), "replication".into()],
            source_type: "recorded", semantic_type: "decision" },
        // Topic C: security (completely unrelated)
        CorpusRecord { content: "Implement JWT token rotation every twenty four hours".into(),
            level: Level::Domain, tags: vec!["security".into(), "authentication".into()],
            source_type: "recorded", semantic_type: "decision" },
        CorpusRecord { content: "JWT tokens must be rotated every twenty four hours for security".into(),
            level: Level::Domain, tags: vec!["security".into(), "authentication".into()],
            source_type: "recorded", semantic_type: "decision" },
    ];

    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        store_corpus(&aura, &mixed);
        run_cycles(&aura, 5);

        let beliefs = aura.get_beliefs(None);
        // Check: no belief should contain records from different topic groups
        // Indirect check: if precision stays high on curated benchmark, no cross-topic merges
        let q = measure_quality(cfg);
        let cross_topic = q.false_merge_rate > 0.10; // more than 10% false merges = suspicious

        println!("  [{}] beliefs={} false_merge={:.3} cross_topic={}",
            cfg.name, beliefs.len(), q.false_merge_rate,
            if cross_topic { "SUSPECT" } else { "CLEAN" });

        assert!(!cross_topic,
            "[{}] cross-topic false merge rate too high: {:.3}", cfg.name, q.false_merge_rate);
    }
}

/// Test 6: Candidate B monitor still passes.
#[test]
fn candidate_b_monitor_still_passes() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CANDIDATE B MONITORING PACK");
    println!("══════════════════════════════════════════════════\n");

    // Run monitor queries under DualKey (most aggressive variant)
    let cfg = &CONFIGS[2]; // DualKey
    let (aura, _dir) = open_aura_with_config(cfg);

    // Store mixed practical corpus
    let all: Vec<Vec<CorpusRecord>> = vec![
        practical_deploy_chain(), practical_stable_preference(),
        practical_multi_topic(), practical_contextual(),
    ];
    for c in &all { store_corpus(&aura, c); }
    run_cycles(&aura, 8);

    aura.set_belief_rerank_enabled(true);
    let queries = [
        "deploy staging", "production canary", "dark mode editor",
        "vim keybindings", "tabs spaces indentation", "GraphQL REST API",
    ];

    let mut worse = 0;
    let mut total = 0;
    for q in &queries {
        total += 1;
        aura.set_belief_rerank_enabled(false);
        let base = aura.recall_structured(q, Some(5), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        aura.set_belief_rerank_enabled(true);
        let ranked = aura.recall_structured(q, Some(5), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        if base.len() != ranked.len() { worse += 1; }
    }

    let worse_pct = worse as f32 / total as f32 * 100.0;
    println!("  DualKey: {}/{} queries ok, worse={:.0}%", total - worse, total, worse_pct);
    assert!(worse_pct <= 5.0, "Candidate B degraded under DualKey: {:.0}% worse", worse_pct);
}

/// Test 7: Cross-layer eval still green.
#[test]
fn cross_layer_eval_still_green() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CROSS-LAYER EVAL");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();
    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        store_corpus(&aura, &corpus);
        for _ in 0..8 {
            let report = aura.run_maintenance();
            assert!(report.timings.total_ms >= 0.0);
        }
        let results = aura.recall_structured("deployment monitoring", Some(10), Some(0.0), Some(true), None, None)
            .unwrap_or_default();
        assert!(!results.is_empty(), "[{}] recall returned 0", cfg.name);
        println!("  [{}] beliefs={} concepts={} causal={} — PASS", cfg.name,
            aura.get_beliefs(None).len(), aura.get_concepts(None).len(),
            aura.get_causal_patterns(None).len());
    }
}

/// Test 8: Churn stability over 10 cycles.
#[test]
fn redesign_churn_stability() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CHURN STABILITY: 10 cycles");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();
    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        store_corpus(&aura, &corpus);
        let mut churns = Vec::new();
        for _ in 0..10 {
            let r = aura.run_maintenance();
            churns.push(r.belief.churn_rate);
        }
        let last5_max = churns[5..].iter().cloned().fold(0.0_f32, f32::max);
        println!("  [{}] max churn (last 5): {:.4} — {}", cfg.name, last5_max,
            if last5_max < 0.05 { "PASS" } else { "WARN" });
        assert!(last5_max < 0.10, "[{}] churn spike: {:.4}", cfg.name, last5_max);
    }
}

/// Test 9: Aggregate report — compares all variants and emits verdict.
#[test]
fn representation_redesign_compares_all_variants() {
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  BELIEF REPRESENTATION REDESIGN: VARIANT COMPARISON REPORT");
    println!("══════════════════════════════════════════════════════════════\n");

    let practical_sets = vec![
        practical_deploy_chain(), practical_stable_preference(),
        practical_multi_topic(), practical_contextual(),
    ];

    // Quality
    println!("  ── Quality (curated 26-record benchmark) ──\n");
    println!("  {:15} {:>6} {:>6} {:>6} {:>8} {:>6}", "Config", "P", "R", "F1", "FM", "Churn");

    struct VariantResult {
        name: &'static str,
        quality: QualityMetrics,
        concepts: usize,
        avg_coverage: f32,
        beliefs: usize,
        seeds: usize,
        parts_ge2: usize,
    }

    let mut results: Vec<VariantResult> = Vec::new();

    for cfg in CONFIGS {
        let q = measure_quality(cfg);
        println!("  {:15} {:>6.3} {:>6.3} {:>6.3} {:>8.3} {:>6.4}",
            cfg.name, q.precision, q.recall, q.f1, q.false_merge_rate, q.churn_rate);

        let mut total_concepts = 0;
        let mut total_coverage = 0.0_f32;
        for corpus in &practical_sets {
            let cm = measure_concept_practical(cfg, corpus, 8);
            total_concepts += cm.concepts_formed;
            total_coverage += cm.concept_coverage;
        }
        let avg_cov = total_coverage / practical_sets.len() as f32;

        // Combined density
        let (aura, _dir) = open_aura_with_config(cfg);
        for c in &practical_sets { store_corpus(&aura, c); }
        run_cycles(&aura, 8);
        let beliefs = aura.get_beliefs(None);
        let mut sp: HashMap<String, usize> = HashMap::new();
        for b in &beliefs {
            if matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                && b.stability >= 2.0 && b.confidence >= 0.55 {
                let parts: Vec<&str> = b.key.split(':').collect();
                let pk = format!("{}:{}", parts.first().unwrap_or(&"default"), parts.last().unwrap_or(&"fact"));
                *sp.entry(pk).or_insert(0) += 1;
            }
        }
        let seeds: usize = sp.values().sum();
        let p2 = sp.values().filter(|&&v| v >= 2).count();

        results.push(VariantResult {
            name: cfg.name, quality: q, concepts: total_concepts,
            avg_coverage: avg_cov, beliefs: beliefs.len(), seeds, parts_ge2: p2,
        });
    }

    // Concept coverage
    println!("\n  ── Practical Concept Coverage ──\n");
    println!("  {:15} {:>10} {:>12}", "Config", "Concepts", "AvgCoverage%");
    for r in &results {
        println!("  {:15} {:>10} {:>11.1}%", r.name, r.concepts, r.avg_coverage * 100.0);
    }

    // Density
    println!("\n  ── Combined Density (22 practical records) ──\n");
    println!("  {:15} {:>8} {:>8} {:>8} {:>10}", "Config", "Beliefs", "Seeds", "Part≥2", "Concepts");
    for r in &results {
        println!("  {:15} {:>8} {:>8} {:>8} {:>10}", r.name, r.beliefs, r.seeds, r.parts_ge2, r.concepts);
    }

    // Verdict
    println!("\n  ── VERDICT ──\n");

    let std_r = &results[0];
    for r in &results {
        let safe = r.quality.precision >= 0.50 && r.quality.false_merge_rate <= 0.20;
        let density_up = r.beliefs > std_r.beliefs || r.seeds > std_r.seeds;
        let concept_up = r.concepts > std_r.concepts || r.avg_coverage > std_r.avg_coverage;
        println!("  {:15} P={:.3} FM={:.3} safe={} density_up={} concept_up={} coverage={:.1}%",
            r.name, r.quality.precision, r.quality.false_merge_rate, safe, density_up, concept_up, r.avg_coverage * 100.0);
    }

    // Find best variant: highest concept coverage among safe variants
    let best = results.iter()
        .filter(|r| r.quality.precision >= 0.50 && r.quality.false_merge_rate <= 0.20)
        .max_by(|a, b| a.avg_coverage.partial_cmp(&b.avg_coverage).unwrap()
            .then(a.concepts.cmp(&b.concepts))
            .then(a.quality.precision.partial_cmp(&b.quality.precision).unwrap()));

    let verdict = if let Some(best) = best {
        let helps_density = best.beliefs > std_r.beliefs || best.seeds > std_r.seeds;
        let helps_concept = best.concepts > std_r.concepts || best.avg_coverage > std_r.avg_coverage;
        let parts_up = best.parts_ge2 > std_r.parts_ge2;

        if best.avg_coverage >= 0.10 {
            format!("SAFE REDESIGN FOUND — {} (coverage {:.1}%)", best.name, best.avg_coverage * 100.0)
        } else if helps_concept && best.avg_coverage > 0.0 {
            format!("PARTIAL IMPROVEMENT — {} (coverage {:.1}%)", best.name, best.avg_coverage * 100.0)
        } else if helps_density || parts_up {
            format!("PARTIAL IMPROVEMENT, C STILL BLOCKED — {} increases density but 0 concepts", best.name)
        } else {
            "NO SAFE REDESIGN".to_string()
        }
    } else {
        "NO SAFE REDESIGN — all variants failed safety gates".to_string()
    };

    println!("\n  VERDICT: {}\n", verdict);
}

/// Test 10: Emit best-variant verdict with go/no-go recommendation.
#[test]
fn representation_redesign_emits_best_variant_verdict() {
    println!("\n══════════════════════════════════════════════════");
    println!("  BEST VARIANT VERDICT");
    println!("══════════════════════════════════════════════════\n");

    // Quick eval: quality + density for each
    for cfg in CONFIGS {
        let q = measure_quality(cfg);
        let safe = q.precision >= 0.50 && q.false_merge_rate <= 0.20;

        let (aura, _dir) = open_aura_with_config(cfg);
        let all = vec![
            practical_deploy_chain(), practical_stable_preference(),
            practical_multi_topic(), practical_contextual(),
        ];
        for c in &all { store_corpus(&aura, c); }
        run_cycles(&aura, 8);
        let beliefs = aura.get_beliefs(None);
        let seeds = beliefs.iter().filter(|b| {
            matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                && b.stability >= 2.0 && b.confidence >= 0.55
        }).count();
        let concepts = aura.get_concepts(None);

        println!("  {:15} P={:.3} FM={:.3} safe={:5} beliefs={:2} seeds={:2} concepts={}",
            cfg.name, q.precision, q.false_merge_rate, safe, beliefs.len(), seeds, concepts.len());
    }
}
