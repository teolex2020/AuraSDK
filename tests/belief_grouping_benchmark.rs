//! Formal grouping precision benchmark for the belief engine.
//!
//! Uses a curated labeled dataset to measure:
//! - Pairwise precision (what fraction of grouped pairs should be grouped)
//! - Pairwise recall (what fraction of should-be-grouped pairs are grouped)
//! - F1 score
//! - False merge rate
//! - False split rate
//!
//! Acceptance criteria:
//! - Precision ≥ 0.85
//! - Recall ≥ 0.70
//! - F1 ≥ 0.75
//! - False merge rate ≤ 0.15
//! - Churn rate < 0.05 on stable replay

use aura::{Aura, Level};

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

fn open_temp_aura() -> (Aura, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    (aura, dir)
}

/// A labeled record descriptor.
struct LabeledRecord {
    content: &'static str,
    level: Level,
    tags: &'static [&'static str],
    source_type: &'static str,
    semantic_type: &'static str,
    /// Cluster label — records with the same cluster_id SHOULD group together.
    cluster_id: u32,
}

/// Store all labeled records and return their IDs in order.
fn store_labeled(aura: &Aura, records: &[LabeledRecord]) -> Vec<String> {
    records.iter().map(|lr| {
        let rec = aura.store(
            lr.content,
            Some(lr.level),
            Some(lr.tags.iter().map(|t| t.to_string()).collect()),
            None, None,
            Some(lr.source_type),
            None,
            Some(false), // disable dedup
            None, None,
            Some(lr.semantic_type),
        ).unwrap_or_else(|e| panic!("store failed for '{}': {}", lr.content, e));
        rec.id.clone()
    }).collect()
}

/// After maintenance, reconstruct belief grouping to check which
/// record pairs ended up in the same belief.
/// Returns a set of (min_id, max_id) pairs that share a belief.
fn grouped_pairs(aura: &Aura, record_ids: &[String]) -> std::collections::HashSet<(String, String)> {
    // Run maintenance to trigger belief update
    let _report = aura.run_maintenance();

    // Reconstruct grouping using the same algorithm as the engine.
    // Use aura.get() to retrieve records by ID (reliable, not recall-dependent).
    use aura::belief::{BeliefEngine, SdrLookup};
    use aura::sdr::SDRInterpreter;

    let mut records_map = std::collections::HashMap::new();
    for rid in record_ids {
        if let Some(rec) = aura.get(rid) {
            records_map.insert(rid.clone(), rec);
        }
    }

    // Build SDR lookup — always use is_identity=false (general bit range)
    // so records at different levels can be compared consistently.
    let sdr = SDRInterpreter::default();
    let mut sdr_lookup: SdrLookup = std::collections::HashMap::new();
    for (rid, rec) in &records_map {
        let sdr_vec = sdr.text_to_sdr(&rec.content, false);
        sdr_lookup.insert(rid.clone(), sdr_vec);
    }

    // Run belief engine on snapshot
    let mut engine = BeliefEngine::new();
    engine.update_with_sdr(&records_map, &sdr_lookup);

    // Extract which pairs share a belief
    let mut pairs = std::collections::HashSet::new();
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
                let pair = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                pairs.insert(pair);
            }
        }
    }

    pairs
}

/// Compute precision/recall/F1 from labeled data and actual grouping.
struct BenchmarkResult {
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    true_negatives: usize,
    precision: f32,
    recall: f32,
    f1: f32,
    false_merge_rate: f32,
    false_split_rate: f32,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TP={} FP={} FN={} TN={} | precision={:.3} recall={:.3} F1={:.3} | false_merge={:.3} false_split={:.3}",
            self.true_positives, self.false_positives,
            self.false_negatives, self.true_negatives,
            self.precision, self.recall, self.f1,
            self.false_merge_rate, self.false_split_rate)
    }
}

fn compute_metrics(
    record_ids: &[String],
    cluster_labels: &[u32],
    actual_pairs: &std::collections::HashSet<(String, String)>,
) -> BenchmarkResult {
    let n = record_ids.len();
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fnn = 0usize;
    let mut tn = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let should_group = cluster_labels[i] == cluster_labels[j];
            let pair = if record_ids[i] < record_ids[j] {
                (record_ids[i].clone(), record_ids[j].clone())
            } else {
                (record_ids[j].clone(), record_ids[i].clone())
            };
            let is_grouped = actual_pairs.contains(&pair);

            match (should_group, is_grouped) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fnn += 1,
                (false, false) => tn += 1,
            }
        }
    }

    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 1.0 };
    let recall = if tp + fnn > 0 { tp as f32 / (tp + fnn) as f32 } else { 1.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    let false_merge_rate = if fp + tn > 0 { fp as f32 / (fp + tn) as f32 } else { 0.0 };
    let false_split_rate = if fnn + tp > 0 { fnn as f32 / (fnn + tp) as f32 } else { 0.0 };

    BenchmarkResult {
        true_positives: tp,
        false_positives: fp,
        false_negatives: fnn,
        true_negatives: tn,
        precision,
        recall,
        f1,
        false_merge_rate,
        false_split_rate,
    }
}

// ═════════════════════════════════════════════════════════
// Curated labeled dataset
// ═════════════════════════════════════════════════════════

fn curated_dataset() -> Vec<LabeledRecord> {
    vec![
        // ── Cluster 1: deployment pipeline (same claim, paraphrases) ──
        LabeledRecord {
            content: "Always deploy through the CI pipeline to staging before production",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1,
        },
        LabeledRecord {
            content: "Deploy through CI pipeline to staging before going to production",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1,
        },
        LabeledRecord {
            content: "CI pipeline must deploy to staging first before production release",
            level: Level::Decisions, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 1,
        },

        // ── Cluster 2: database config (different topic, shares "config" intent) ──
        LabeledRecord {
            content: "Set PostgreSQL connection pool maximum to thirty active connections",
            level: Level::Decisions, tags: &["database", "config", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 2,
        },
        LabeledRecord {
            content: "PostgreSQL connection pool maximum should be thirty connections",
            level: Level::Decisions, tags: &["database", "config", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 2,
        },

        // ── Cluster 3: coding preferences (user style) ──
        LabeledRecord {
            content: "I prefer dark mode with high contrast theme in all code editors",
            level: Level::Domain, tags: &["ui", "preferences", "coding"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 3,
        },
        LabeledRecord {
            content: "Dark mode with high contrast is my preferred coding editor theme",
            level: Level::Domain, tags: &["ui", "preferences", "coding"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 3,
        },

        // ── Cluster 4: reading preferences (different context, shares "ui"/"preferences") ──
        LabeledRecord {
            content: "For reading documentation I prefer light mode with serif font",
            level: Level::Domain, tags: &["ui", "preferences", "reading"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 4,
        },
        LabeledRecord {
            content: "Light mode with serif fonts works best for reading documentation",
            level: Level::Domain, tags: &["ui", "preferences", "reading"],
            source_type: "recorded", semantic_type: "preference", cluster_id: 4,
        },

        // ── Cluster 5: Rust backend fact ──
        LabeledRecord {
            content: "The entire backend service layer is written in Rust for performance",
            level: Level::Identity, tags: &["tech", "rust", "backend"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 5,
        },
        LabeledRecord {
            content: "Our backend services are all implemented in Rust for performance",
            level: Level::Identity, tags: &["tech", "rust", "backend"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 5,
        },

        // ── Cluster 6: CI automation (different topic, shares "ci" with cluster 1) ──
        LabeledRecord {
            content: "GitHub Actions runs all continuous integration workflows automatically",
            level: Level::Domain, tags: &["tech", "ci", "automation"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 6,
        },
        LabeledRecord {
            content: "All CI workflows are automated through GitHub Actions pipelines",
            level: Level::Domain, tags: &["tech", "ci", "automation"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 6,
        },

        // ── Cluster 7: API architecture decision ──
        LabeledRecord {
            content: "We use gRPC for all latency-critical inter-service communication",
            level: Level::Decisions, tags: &["architecture", "api", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 7,
        },
        LabeledRecord {
            content: "gRPC is the standard for latency-critical service communication",
            level: Level::Decisions, tags: &["architecture", "api", "performance"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 7,
        },

        // ── Cluster 8: logging config (different topic, shares "config") ──
        LabeledRecord {
            content: "Production logging verbosity is set to INFO level for all services",
            level: Level::Domain, tags: &["logging", "config", "operations"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 8,
        },
        LabeledRecord {
            content: "All services use INFO log level in production environment",
            level: Level::Domain, tags: &["logging", "config", "operations"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 8,
        },

        // ── Cluster 9: testing policy ──
        LabeledRecord {
            content: "Integration tests must pass before any pull request can be merged",
            level: Level::Decisions, tags: &["testing", "process", "quality"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 9,
        },
        LabeledRecord {
            content: "No pull request merge without passing integration test suite first",
            level: Level::Decisions, tags: &["testing", "process", "quality"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 9,
        },

        // ── Cluster 10: security — infosec (different from physical security) ──
        LabeledRecord {
            content: "Enable two-factor authentication for all production admin accounts",
            level: Level::Decisions, tags: &["security", "auth", "production"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 10,
        },
        LabeledRecord {
            content: "All production admin accounts require two-factor authentication",
            level: Level::Decisions, tags: &["security", "auth", "production"],
            source_type: "recorded", semantic_type: "decision", cluster_id: 10,
        },

        // ── Cluster 11: contradiction within deployment (same claim as 1, opposing) ──
        LabeledRecord {
            content: "Sometimes we skip staging and deploy hotfixes direct to production",
            level: Level::Working, tags: &["deploy", "ci", "process"],
            source_type: "recorded", semantic_type: "contradiction", cluster_id: 1,
        },

        // ── Cluster 12: Ukrainian language records (non-ASCII) ──
        LabeledRecord {
            content: "Використовуємо Kubernetes для оркестрації всіх мікросервісів",
            level: Level::Domain, tags: &["infra", "kubernetes", "orchestration"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 12,
        },
        LabeledRecord {
            content: "Kubernetes оркеструє всі наші мікросервіси в продакшені",
            level: Level::Domain, tags: &["infra", "kubernetes", "orchestration"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 12,
        },

        // ── Cluster 13: rate limiting (shares "api" with cluster 7) ──
        LabeledRecord {
            content: "API rate limiting is configured at one thousand requests per minute",
            level: Level::Domain, tags: &["api", "config", "performance"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 13,
        },
        LabeledRecord {
            content: "Rate limiting set to one thousand requests per minute for all APIs",
            level: Level::Domain, tags: &["api", "config", "performance"],
            source_type: "recorded", semantic_type: "fact", cluster_id: 13,
        },
    ]
}

// ═════════════════════════════════════════════════════════
// Benchmark tests
// ═════════════════════════════════════════════════════════

/// Main precision/recall/F1 benchmark on the curated dataset.
#[test]
fn benchmark_grouping_precision_recall() {
    let (aura, _dir) = open_temp_aura();
    let dataset = curated_dataset();
    let cluster_labels: Vec<u32> = dataset.iter().map(|r| r.cluster_id).collect();
    let record_ids = store_labeled(&aura, &dataset);

    assert_eq!(record_ids.len(), dataset.len(), "all records should be stored");

    // Run 3 maintenance cycles to stabilize
    for _ in 0..3 {
        aura.run_maintenance();
    }

    let actual_pairs = grouped_pairs(&aura, &record_ids);

    let result = compute_metrics(&record_ids, &cluster_labels, &actual_pairs);

    eprintln!("\n=== GROUPING BENCHMARK ===");
    eprintln!("Dataset: {} records, {} clusters", dataset.len(), {
        let mut ids: Vec<u32> = cluster_labels.clone();
        ids.sort();
        ids.dedup();
        ids.len()
    });
    eprintln!("Pairs: {} total, {} grouped", {
        let n = record_ids.len();
        n * (n - 1) / 2
    }, actual_pairs.len());
    eprintln!("{}", result);
    eprintln!("==========================\n");

    // ── Acceptance criteria ──
    assert!(
        result.precision >= 0.85,
        "PRECISION {:.3} < 0.85 — too many false merges (FP={})",
        result.precision, result.false_positives
    );
    assert!(
        result.recall >= 0.70,
        "RECALL {:.3} < 0.70 — too many false splits (FN={})",
        result.recall, result.false_negatives
    );
    assert!(
        result.f1 >= 0.75,
        "F1 {:.3} < 0.75",
        result.f1
    );
    assert!(
        result.false_merge_rate <= 0.15,
        "FALSE MERGE RATE {:.3} > 0.15",
        result.false_merge_rate
    );
}

/// Churn benchmark: stable dataset through 10 cycles, churn should converge < 0.05.
#[test]
fn benchmark_churn_convergence() {
    let (aura, _dir) = open_temp_aura();
    let dataset = curated_dataset();
    let _record_ids = store_labeled(&aura, &dataset);

    // Run 10 maintenance cycles
    let mut reports = Vec::new();
    for _ in 0..10 {
        reports.push(aura.run_maintenance());
    }

    let churn_rates: Vec<f32> = reports.iter().map(|r| r.belief.churn_rate).collect();
    eprintln!("\n=== CHURN CONVERGENCE ===");
    for (i, rate) in churn_rates.iter().enumerate() {
        eprintln!("  cycle {}: churn={:.4}", i, rate);
    }
    eprintln!("========================\n");

    // Last 5 cycles should have churn < 0.05
    let last_5 = &churn_rates[5..];
    let avg_churn: f32 = last_5.iter().sum::<f32>() / last_5.len() as f32;

    assert!(
        avg_churn < 0.05,
        "average churn in last 5 cycles = {:.4}, expected < 0.05 — beliefs not converging\n\
         full churn trace: {:?}",
        avg_churn, churn_rates
    );

    // No cycle in last 5 should exceed 0.10
    for (i, &rate) in last_5.iter().enumerate() {
        assert!(
            rate < 0.10,
            "cycle {} churn = {:.4} exceeds 0.10",
            i + 5, rate
        );
    }
}

/// Per-cluster precision: verify each cluster individually to identify weak spots.
#[test]
fn benchmark_per_cluster_accuracy() {
    let (aura, _dir) = open_temp_aura();
    let dataset = curated_dataset();
    let cluster_labels: Vec<u32> = dataset.iter().map(|r| r.cluster_id).collect();
    let record_ids = store_labeled(&aura, &dataset);

    for _ in 0..3 {
        aura.run_maintenance();
    }

    let actual_pairs = grouped_pairs(&aura, &record_ids);

    // Check each cluster: all records with the same cluster_id should be grouped
    let mut cluster_ids: Vec<u32> = cluster_labels.clone();
    cluster_ids.sort();
    cluster_ids.dedup();

    eprintln!("\n=== PER-CLUSTER ACCURACY ===");
    let mut failed_clusters = Vec::new();

    for &cid in &cluster_ids {
        let members: Vec<usize> = cluster_labels.iter().enumerate()
            .filter(|(_, &label)| label == cid)
            .map(|(i, _)| i)
            .collect();

        if members.len() < 2 {
            continue; // singleton cluster — nothing to check
        }

        let mut intra_grouped = 0;
        let mut intra_total = 0;

        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a = &record_ids[members[i]];
                let b = &record_ids[members[j]];
                let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                intra_total += 1;
                if actual_pairs.contains(&pair) {
                    intra_grouped += 1;
                }
            }
        }

        let intra_recall = intra_grouped as f32 / intra_total as f32;
        let status = if intra_recall >= 0.50 { "OK" } else { "WEAK" };
        eprintln!("  cluster {} (n={}): {}/{} grouped (recall={:.2}) [{}]",
            cid, members.len(), intra_grouped, intra_total, intra_recall, status);

        if intra_recall < 0.50 {
            failed_clusters.push(cid);
        }
    }
    eprintln!("============================\n");

    // At most 2 clusters can have weak recall (due to SDR paraphrase limits)
    assert!(
        failed_clusters.len() <= 2,
        "too many clusters with weak recall (<0.50): {:?}",
        failed_clusters
    );
}
