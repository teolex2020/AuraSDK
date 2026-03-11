//! Belief Grouping Densification Sprint
//!
//! Tests 3 coarse key variants against baseline to find a safe way to increase
//! belief partition density, unblocking Candidate C (concept-assisted grouping)
//! on practical corpora.
//!
//! Variants:
//!   Standard  — namespace:sorted_tags(top3):semantic_type (baseline)
//!   TopOneTag — namespace:top_1_sorted_tag:semantic_type
//!   SemanticOnly — namespace:semantic_type (no tags; SDR does all fine grouping)
//!
//! Measures: belief density, singleton rate, precision/recall/F1,
//! concept seed count, concept coverage on practical packs,
//! Candidate B safety (rerank not regressed).

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

fn open_temp_aura_with_mode(mode: CoarseKeyMode) -> (Aura, tempfile::TempDir) {
    let (aura, dir) = open_temp_aura();
    aura.set_belief_coarse_key_mode(mode);
    (aura, dir)
}

fn run_cycles(aura: &Aura, n: usize) {
    for _ in 0..n {
        aura.run_maintenance();
    }
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
            Some(lr.level),
            Some(lr.tags.iter().map(|t| t.to_string()).collect()),
            None, None,
            Some(lr.source_type),
            None,
            Some(false),
            None, None,
            Some(lr.semantic_type),
        ).unwrap_or_else(|e| panic!("store failed for '{}': {}", lr.content, e));
        rec.id.clone()
    }).collect()
}

fn curated_dataset() -> Vec<LabeledRecord> {
    vec![
        // Cluster 1: deployment pipeline
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
        // Cluster 11 → contradiction within cluster 1
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
// Practical corpora (from concept_shadow_eval)
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

#[derive(Debug, Clone, Copy)]
struct DensifyConfig {
    name: &'static str,
    coarse_mode: CoarseKeyMode,
    threshold_override: Option<f32>,
}

const CONFIGS: &[DensifyConfig] = &[
    DensifyConfig { name: "Standard", coarse_mode: CoarseKeyMode::Standard, threshold_override: None },
    DensifyConfig { name: "TopOneTag", coarse_mode: CoarseKeyMode::TopOneTag, threshold_override: None },
    DensifyConfig { name: "SemanticOnly", coarse_mode: CoarseKeyMode::SemanticOnly, threshold_override: None },
    DensifyConfig { name: "Semantic+0.10", coarse_mode: CoarseKeyMode::SemanticOnly, threshold_override: Some(0.10) },
];

fn open_aura_with_config(cfg: &DensifyConfig) -> (Aura, tempfile::TempDir) {
    let (aura, dir) = open_temp_aura();
    aura.set_belief_coarse_key_mode(cfg.coarse_mode);
    if let Some(t) = cfg.threshold_override {
        aura.set_belief_similarity_threshold(Some(t));
    }
    (aura, dir)
}

// ═══════════════════════════════════════════════════════════
// Measurement helpers
// ═══════════════════════════════════════════════════════════

/// Measure precision/recall/F1 using the curated benchmark dataset.
fn measure_quality(cfg: &DensifyConfig) -> QualityMetrics {
    let (aura, _dir) = open_aura_with_config(cfg);
    let dataset = curated_dataset();
    let cluster_labels: Vec<u32> = dataset.iter().map(|r| r.cluster_id).collect();
    let record_ids = store_labeled(&aura, &dataset);

    // Run maintenance
    let _report = aura.run_maintenance();

    // Reconstruct grouping using direct belief engine with same mode
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

    // Extract pairs that share a belief
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

    // Compute pairwise metrics
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

    // Measure churn over 5 cycles
    let mut engine2 = BeliefEngine::with_coarse_key_mode(cfg.coarse_mode);
    engine2.claim_similarity_override = cfg.threshold_override;
    let mut last_churn = 0.0_f32;
    for _ in 0..5 {
        let report = engine2.update_with_sdr(&records_map, &sdr_lookup);
        last_churn = report.churn_rate;
    }

    QualityMetrics { precision, recall, f1, false_merge_rate, false_split_rate, churn_rate: last_churn }
}

/// Measure concept formation on a practical corpus with given config.
fn measure_concept_practical(cfg: &DensifyConfig, corpus: &[CorpusRecord], cycles: usize) -> ConceptMetrics {
    let (aura, _dir) = open_aura_with_config(cfg);
    store_corpus(&aura, corpus);
    run_cycles(&aura, cycles);

    let concepts = aura.get_concepts(None);
    let stable = concepts.iter().filter(|c| c.state == ConceptState::Stable).count();

    // Measure coverage: query each record's content, check if any result has concept membership
    let mut covered = 0usize;
    let mut total = 0usize;
    for rec in corpus {
        total += 1;
        let results = aura.recall_structured(
            &rec.content, Some(5), Some(0.0), Some(true), None, None,
        ).unwrap_or_default();
        if !results.is_empty() {
            // Check if any returned record belongs to a concept
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

/// Test 0: Quick diagnostic — verify mode actually changes belief grouping.
#[test]
fn densification_mode_diagnostic() {
    let (aura, _dir) = open_temp_aura_with_mode(CoarseKeyMode::SemanticOnly);
    println!("\n  MODE: {:?}", aura.get_belief_coarse_key_mode());

    // 3 decision records with different tags
    aura.store("Deployed version 2.3 to staging environment", Some(Level::Domain),
        Some(vec!["deploy".into(), "staging".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();
    aura.store("Promoted staging build to production canary", Some(Level::Domain),
        Some(vec!["deploy".into(), "production".into(), "canary".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();
    aura.store("Completed full production rollout of version 2.3", Some(Level::Domain),
        Some(vec!["deploy".into(), "production".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();

    // Check what keys the engine would generate by reading records directly
    {
        let rec = aura.store("test dummy for key check that is long enough for belief engine", Some(Level::Domain),
            Some(vec!["deploy".into(), "staging".into()]), None, None,
            Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();
        let key_std = BeliefEngine::claim_key(&rec);
        let key_sem = BeliefEngine::claim_key_with_mode(&rec, CoarseKeyMode::SemanticOnly);
        println!("  Sample: tags={:?} semantic_type={}", rec.tags, rec.semantic_type);
        println!("  Key(Standard):     '{}'", key_std);
        println!("  Key(SemanticOnly): '{}'", key_sem);
        println!("  is_alive={}, content.len={}", rec.is_alive(), rec.content.len());
    }

    // Check Tanimoto between stored records
    {
        let sdr = SDRInterpreter::default();
        let texts = vec![
            "Deployed version 2.3 to staging environment",
            "Promoted staging build to production canary",
            "Completed full production rollout of version 2.3",
            "test dummy for key check that is long enough for belief engine",
        ];
        let sdrs: Vec<Vec<u16>> = texts.iter().map(|t| sdr.text_to_sdr(t, false)).collect();
        for i in 0..sdrs.len() {
            for j in (i+1)..sdrs.len() {
                let intersection = sdrs[i].iter().filter(|b| sdrs[j].contains(b)).count();
                let union = sdrs[i].len() + sdrs[j].len() - intersection;
                let tanimoto = if union == 0 { 0.0 } else { intersection as f32 / union as f32 };
                println!("  Tanimoto({}, {}): {:.4} (SDR lens: {}, {})",
                    i, j, tanimoto, sdrs[i].len(), sdrs[j].len());
            }
        }
    }

    // Run maintenance and check total records
    let report = aura.run_maintenance();
    println!("  After 1st maintenance: total_records={} belief.total={} belief.created={}",
        report.total_records, report.belief.total_beliefs, report.belief.beliefs_created);

    for i in 1..5 {
        let report = aura.run_maintenance();
        println!("  Cycle {}: created={} total={} resolved={} singleton={}",
            i, report.belief.beliefs_created, report.belief.total_beliefs,
            report.belief.resolved, report.belief.total_beliefs - report.belief.resolved);
    }

    let beliefs = aura.get_beliefs(None);
    println!("  Final beliefs (threshold=0.15): {}", beliefs.len());

    // Now try with lowered threshold (0.10) to match concept retune range
    let (aura2, _dir2) = open_temp_aura_with_mode(CoarseKeyMode::SemanticOnly);
    aura2.set_belief_similarity_threshold(Some(0.10));

    aura2.store("Deployed version 2.3 to staging environment", Some(Level::Domain),
        Some(vec!["deploy".into(), "staging".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();
    aura2.store("Promoted staging build to production canary", Some(Level::Domain),
        Some(vec!["deploy".into(), "production".into(), "canary".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();
    aura2.store("Completed full production rollout of version 2.3", Some(Level::Domain),
        Some(vec!["deploy".into(), "production".into()]), None, None,
        Some("recorded"), None, Some(false), None, None, Some("decision")).unwrap();

    for i in 0..5 {
        let report = aura2.run_maintenance();
        println!("  [threshold=0.10] Cycle {}: created={} total={} resolved={}",
            i, report.belief.beliefs_created, report.belief.total_beliefs, report.belief.resolved);
    }

    let beliefs2 = aura2.get_beliefs(None);
    println!("  Final beliefs (threshold=0.10): {}", beliefs2.len());
    for b in &beliefs2 {
        println!("    key={} state={:?} stability={:.1} confidence={:.3}",
            b.key, b.state, b.stability, b.confidence);
    }

    // With SemanticOnly + threshold=0.10, records with Tanimoto ~0.13 should cluster
    assert!(!beliefs2.is_empty(),
        "SemanticOnly + threshold=0.10 should produce beliefs from 3 deploy records (Tanimoto ~0.13)");
}

/// Test 1: Density diagnostics per config on deploy-chain corpus.
#[test]
fn densification_partition_density_per_variant() {
    println!("\n══════════════════════════════════════════════════");
    println!("  DENSITY DIAGNOSTICS: deploy-chain corpus (6 records)");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();

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

        println!("  [{}] beliefs={} seeds={} concepts={}",
            cfg.name, beliefs.len(), seeds, concepts.len());
    }

    // Semantic+0.10 should produce beliefs from deploy records (Tanimoto ~0.13)
    let sem10 = &CONFIGS[3]; // Semantic+0.10
    let (aura, _dir) = open_aura_with_config(sem10);
    store_corpus(&aura, &corpus);
    run_cycles(&aura, 5);
    let beliefs = aura.get_beliefs(None);
    assert!(!beliefs.is_empty(),
        "Semantic+0.10 should produce beliefs on deploy-chain corpus");
    println!("\n  Semantic+0.10 produces {} beliefs — PASS", beliefs.len());
}

/// Test 2: Quality benchmark per config on curated labeled dataset.
#[test]
fn densification_quality_benchmark_per_variant() {
    println!("\n══════════════════════════════════════════════════");
    println!("  QUALITY BENCHMARK: curated dataset (26 records, 13 clusters)");
    println!("══════════════════════════════════════════════════\n");

    for cfg in CONFIGS {
        let quality = measure_quality(cfg);
        println!("  [{}] precision={:.3} recall={:.3} F1={:.3} false_merge={:.3}",
            cfg.name, quality.precision, quality.recall, quality.f1, quality.false_merge_rate);
    }

    // Standard baseline must still pass
    let std_q = measure_quality(&CONFIGS[0]);
    assert!(std_q.precision >= 0.85 && std_q.recall >= 0.70 && std_q.f1 >= 0.75,
        "Standard baseline regressed");
}

/// Test 3: Concept coverage on practical corpora per config.
#[test]
fn densification_concept_practical_coverage() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CONCEPT PRACTICAL COVERAGE: per config");
    println!("══════════════════════════════════════════════════\n");

    let practical_sets: Vec<(&str, Vec<CorpusRecord>)> = vec![
        ("deploy-chain", practical_deploy_chain()),
        ("stable-preference", practical_stable_preference()),
        ("multi-topic", practical_multi_topic()),
        ("contextual", practical_contextual()),
    ];

    for cfg in CONFIGS {
        println!("  [{}]", cfg.name);
        let mut total_concepts = 0;
        let mut total_coverage = 0.0_f32;
        for (name, corpus) in &practical_sets {
            let metrics = measure_concept_practical(cfg, corpus, 8);
            println!("    {}: concepts={} coverage={:.1}%",
                name, metrics.concepts_formed, metrics.concept_coverage * 100.0);
            total_concepts += metrics.concepts_formed;
            total_coverage += metrics.concept_coverage;
        }
        println!("    TOTAL: concepts={} avg_coverage={:.1}%\n",
            total_concepts, total_coverage / practical_sets.len() as f32 * 100.0);
    }
}

/// Test 4: Candidate B safety — reranking bounded under all configs.
#[test]
fn densification_candidate_b_not_regressed() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CANDIDATE B SAFETY: rerank under each config");
    println!("══════════════════════════════════════════════════\n");

    let corpus = practical_deploy_chain();

    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        aura.set_belief_rerank_enabled(true);
        store_corpus(&aura, &corpus);
        run_cycles(&aura, 5);

        let queries = ["deployment to staging", "production canary rollout", "monitoring after deploy"];
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

/// Test 5: Cross-layer stack intact under all configs.
#[test]
fn densification_cross_layer_stack_intact() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CROSS-LAYER STACK: per config");
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
        assert!(!results.is_empty(), "[{}] recall returned 0 results", cfg.name);
        println!("  [{}] beliefs={} concepts={} causal={} — PASS", cfg.name,
            aura.get_beliefs(None).len(), aura.get_concepts(None).len(),
            aura.get_causal_patterns(None).len());
    }
}

/// Test 6: Churn stability — 10 cycles per config.
#[test]
fn densification_churn_stability() {
    println!("\n══════════════════════════════════════════════════");
    println!("  CHURN STABILITY: 10 cycles per config");
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
    }
}

/// Test 7: Combined density — all 4 practical corpora in one brain.
#[test]
fn densification_combined_practical_density() {
    println!("\n══════════════════════════════════════════════════");
    println!("  COMBINED PRACTICAL DENSITY: all 22 records, 8 cycles");
    println!("══════════════════════════════════════════════════\n");

    let all_corpora = vec![
        practical_deploy_chain(), practical_stable_preference(),
        practical_multi_topic(), practical_contextual(),
    ];

    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        for c in &all_corpora { store_corpus(&aura, c); }
        run_cycles(&aura, 8);
        let beliefs = aura.get_beliefs(None);
        let concepts = aura.get_concepts(None);
        let singletons = beliefs.iter().filter(|b| b.state == BeliefState::Singleton).count();

        // Seeds and partitions
        let mut seeds_per_part: HashMap<String, usize> = HashMap::new();
        for b in &beliefs {
            if matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                && b.stability >= 2.0 && b.confidence >= 0.55 {
                let parts: Vec<&str> = b.key.split(':').collect();
                let pk = format!("{}:{}", parts.first().unwrap_or(&"default"), parts.last().unwrap_or(&"fact"));
                *seeds_per_part.entry(pk).or_insert(0) += 1;
            }
        }
        let seeds: usize = seeds_per_part.values().sum();
        let p2 = seeds_per_part.values().filter(|&&v| v >= 2).count();

        println!("  [{}] beliefs={} singletons={} seeds={} part≥2={} concepts={}",
            cfg.name, beliefs.len(), singletons, seeds, p2, concepts.len());
    }
}

/// Test 8: Aggregate report with verdict.
#[test]
fn densification_report_emits_variant_comparison() {
    println!("\n══════════════════════════════════════════════════════════");
    println!("  BELIEF DENSIFICATION: VARIANT COMPARISON REPORT");
    println!("══════════════════════════════════════════════════════════\n");

    let practical_sets = vec![
        practical_deploy_chain(), practical_stable_preference(),
        practical_multi_topic(), practical_contextual(),
    ];

    // Quality
    println!("  ── Quality (curated 26-record benchmark) ──\n");
    println!("  {:15} {:>9} {:>9} {:>9} {:>12}", "Config", "Precision", "Recall", "F1", "FalseMerge");

    let mut results: Vec<(&str, QualityMetrics, usize, f32)> = Vec::new();
    for cfg in CONFIGS {
        let q = measure_quality(cfg);
        // Concept coverage
        let mut total_concepts = 0;
        let mut total_coverage = 0.0_f32;
        for corpus in &practical_sets {
            let cm = measure_concept_practical(cfg, corpus, 8);
            total_concepts += cm.concepts_formed;
            total_coverage += cm.concept_coverage;
        }
        let avg_cov = total_coverage / practical_sets.len() as f32;
        println!("  {:15} {:>9.3} {:>9.3} {:>9.3} {:>12.3}",
            cfg.name, q.precision, q.recall, q.f1, q.false_merge_rate);
        results.push((cfg.name, q, total_concepts, avg_cov));
    }

    // Concept coverage
    println!("\n  ── Practical Concept Coverage ──\n");
    println!("  {:15} {:>10} {:>12}", "Config", "Concepts", "AvgCoverage%");
    for r in &results {
        println!("  {:15} {:>10} {:>11.1}%", r.0, r.2, r.3 * 100.0);
    }

    // Combined density
    println!("\n  ── Combined Density ──\n");
    println!("  {:15} {:>8} {:>8} {:>10} {:>10}", "Config", "Beliefs", "Seeds", "Part≥2", "Concepts");
    for cfg in CONFIGS {
        let (aura, _dir) = open_aura_with_config(cfg);
        for c in &practical_sets { store_corpus(&aura, c); }
        run_cycles(&aura, 8);
        let beliefs = aura.get_beliefs(None);
        let concepts = aura.get_concepts(None);
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
        println!("  {:15} {:>8} {:>8} {:>10} {:>10}", cfg.name, beliefs.len(), seeds, p2, concepts.len());
    }

    // Verdict
    println!("\n  ── VERDICT ──\n");

    let std_r = &results[0];
    let best_idx = results.iter().enumerate()
        .max_by(|(_, a), (_, b)| {
            // Score: concept coverage first, then precision as tiebreak
            a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.precision.partial_cmp(&b.1.precision).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(i, _)| i).unwrap_or(0);

    let best = &results[best_idx];
    let best_safe = best.1.precision >= 0.50 && best.1.false_merge_rate <= 0.20;
    let best_helps = best.2 > std_r.2 || best.3 > std_r.3;

    let verdict = if !best_helps || !best_safe {
        "NO SAFE DENSIFICATION"
    } else if best.3 >= 0.10 {
        "DENSIFICATION ENABLES C SHADOW UPGRADE"
    } else if best.3 > 0.0 {
        "SAFE DENSIFICATION FOUND"
    } else if best.2 > std_r.2 {
        "DENSIFICATION HELPS BUT C STILL BLOCKED"
    } else {
        "NO SAFE DENSIFICATION"
    };

    for r in &results {
        let safe = r.1.precision >= 0.50 && r.1.false_merge_rate <= 0.20;
        println!("  {:15} precision={:.3} concepts={} coverage={:.1}% safe={}",
            r.0, r.1.precision, r.2, r.3 * 100.0, safe);
    }
    println!();
    println!("  VERDICT:      {}", verdict);
    println!("  BEST CONFIG:  {}", best.0);
    println!();
}
