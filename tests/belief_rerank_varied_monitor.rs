//! Varied synthetic monitoring for Candidate B.
//!
//! Unlike `belief_rerank_monitor.rs`, this harness intentionally perturbs the
//! corpus and query wording across multiple deterministic runs to test whether
//! belief-aware reranking remains stable under small data variations.

use aura::{Aura, Level};
use std::collections::HashSet;
use std::mem::ManuallyDrop;

const RUNS: usize = 10;
const ALERT_MAX_WORSE_PCT: f32 = 0.05;
const ALERT_MIN_AVG_OVERLAP: f32 = 0.70;
const ALERT_MAX_POS_SHIFT: usize = 2;
const ALERT_MAX_AVG_LATENCY_US: f64 = 2000.0;

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn pick<'a>(&mut self, items: &'a [&'a str]) -> &'a str {
        let idx = (self.next() as usize) % items.len();
        items[idx]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum QualityLabel {
    Better,
    Same,
    Worse,
    Unclear,
}

#[derive(Debug)]
struct QueryMetrics {
    query: String,
    top_k_overlap: f32,
    records_moved: usize,
    max_up_shift: usize,
    max_down_shift: usize,
    belief_coverage: f32,
    latency_delta_us: i64,
    rerank_was_applied: bool,
    contradiction_delta: i32,
    label: QualityLabel,
}

#[derive(Debug)]
struct RunReport {
    reranked_pct: f32,
    better_pct: f32,
    same_pct: f32,
    worse_pct: f32,
    avg_overlap: f32,
    avg_coverage: f32,
    avg_latency_delta_us: f64,
    max_up_shift: usize,
    max_down_shift: usize,
}

struct TempAura {
    aura: ManuallyDrop<Aura>,
    dir: ManuallyDrop<tempfile::TempDir>,
}

impl std::ops::Deref for TempAura {
    type Target = Aura;

    fn deref(&self) -> &Self::Target {
        &self.aura
    }
}

impl Drop for TempAura {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.aura);
            ManuallyDrop::drop(&mut self.dir);
        }
    }
}

fn open_temp_aura() -> TempAura {
    let dir = tempfile::tempdir().expect("tempdir");
    let aura = Aura::open(dir.path().to_str().unwrap()).expect("open aura");
    TempAura {
        aura: ManuallyDrop::new(aura),
        dir: ManuallyDrop::new(dir),
    }
}

fn store(aura: &Aura, content: String, level: Level, tags: &[&str], semantic: &str) {
    aura.store(
        &content,
        Some(level),
        Some(tags.iter().map(|t| t.to_string()).collect()),
        None,
        None,
        Some("recorded"),
        None,
        Some(false),
        None,
        None,
        Some(semantic),
    )
    .unwrap();
}

fn build_varied_corpus(aura: &Aura, seed: u64) {
    let mut rng = Lcg::new(seed);

    let rust_adj = ["safe", "fast", "reliable", "predictable"];
    let py_adj = ["expressive", "approachable", "flexible", "productive"];
    let deploy_adj = ["safe", "controlled", "reliable", "gradual"];
    let db_adj = ["relational", "document", "cached", "embedded"];
    let workflow_adj = ["focused", "disciplined", "clean", "repeatable"];

    for i in 0..6 {
        let content = format!(
            "Rust is {} for backend systems and memory safety {}",
            rng.pick(&rust_adj),
            i
        );
        store(aura, content, Level::Domain, &["rust", "programming", "backend"], "fact");
    }
    for i in 0..5 {
        let content = format!(
            "Python is {} for prototyping and data workflows {}",
            rng.pick(&py_adj),
            i
        );
        store(aura, content, Level::Domain, &["python", "programming", "data"], "fact");
    }
    for i in 0..7 {
        let content = format!(
            "Canary deployment is {} for production rollouts {}",
            rng.pick(&deploy_adj),
            i
        );
        store(aura, content, Level::Domain, &["deploy", "canary", "ops"], "decision");
    }
    for i in 0..6 {
        let content = format!(
            "Database choice should match {} workload characteristics {}",
            rng.pick(&db_adj),
            i
        );
        store(aura, content, Level::Domain, &["database", "choice", "performance"], "fact");
    }
    for i in 0..5 {
        let content = format!(
            "A {} workflow with dark mode and code review improves productivity {}",
            rng.pick(&workflow_adj),
            i
        );
        store(aura, content, Level::Domain, &["workflow", "productivity", "preference"], "preference");
    }

    // Conflicting slices for unresolved belief pressure.
    store(
        aura,
        "Tabs are better than spaces for indentation".to_string(),
        Level::Domain,
        &["formatting", "indentation", "preference"],
        "preference",
    );
    store(
        aura,
        "Spaces are better than tabs for indentation".to_string(),
        Level::Domain,
        &["formatting", "indentation", "preference"],
        "preference",
    );
    store(
        aura,
        "REST is simpler than GraphQL for many APIs".to_string(),
        Level::Domain,
        &["api", "rest", "graphql"],
        "fact",
    );
    store(
        aura,
        "GraphQL reduces over-fetching in frontend APIs".to_string(),
        Level::Domain,
        &["api", "rest", "graphql"],
        "fact",
    );
}

fn varied_queries(seed: u64) -> Vec<(String, usize)> {
    let mut rng = Lcg::new(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let backend = ["backend systems", "backend services", "server-side systems"];
    let deploy = ["production rollout", "deployment strategy", "safe deploy"];
    let db = ["database choice", "relational database", "document workload"];
    let workflow = ["dark mode workflow", "developer workflow", "code review workflow"];
    let conflict = ["tabs vs spaces", "REST vs GraphQL", "indentation preference"];

    vec![
        (format!("Rust {} memory safety", rng.pick(&backend)), 10),
        (format!("Python prototyping {}", rng.pick(&backend)), 10),
        (format!("{} for production", rng.pick(&deploy)), 10),
        (format!("{} performance", rng.pick(&db)), 10),
        (format!("{} productivity", rng.pick(&workflow)), 10),
        (format!("{} API design", rng.pick(&conflict)), 10),
        ("gardening tips for spring".to_string(), 5),
        ("quantum blockchain orchestration".to_string(), 5),
    ]
}

fn compute_quality_label(
    overlap: f32,
    was_applied: bool,
    records_moved: usize,
    contradiction_delta: i32,
    belief_coverage: f32,
) -> QualityLabel {
    if !was_applied || records_moved == 0 {
        return QualityLabel::Same;
    }
    if contradiction_delta > 0 || overlap < 0.5 {
        return QualityLabel::Worse;
    }
    if contradiction_delta < 0 || (belief_coverage >= 0.10 && overlap >= 0.80) {
        return QualityLabel::Better;
    }
    QualityLabel::Unclear
}

fn collect_metrics(aura: &Aura, queries: &[(String, usize)]) -> Vec<QueryMetrics> {
    queries
        .iter()
        .map(|(query, top_k)| {
            let baseline_start = std::time::Instant::now();
            let (baseline, shadow_report) = aura
                .recall_structured_with_shadow(query, Some(*top_k), Some(0.0), Some(true), None, None)
                .unwrap();
            let baseline_latency = baseline_start.elapsed().as_micros() as u64;

            let limited_start = std::time::Instant::now();
            let (limited, rerank_report) = aura
                .recall_structured_with_rerank_report(query, Some(*top_k), Some(0.0), Some(true), None, None)
                .unwrap();
            let limited_latency = limited_start.elapsed().as_micros() as u64;

            let baseline_ids: Vec<String> = baseline.iter().map(|(_, r)| r.id.clone()).collect();
            let limited_ids: Vec<String> = limited.iter().map(|(_, r)| r.id.clone()).collect();
            let effective_k = baseline_ids.len().min(*top_k);

            let baseline_top: HashSet<&str> =
                baseline_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
            let limited_top: HashSet<&str> =
                limited_ids.iter().take(effective_k).map(|s| s.as_str()).collect();
            let overlap = if effective_k > 0 {
                baseline_top.intersection(&limited_top).count() as f32 / effective_k as f32
            } else {
                1.0
            };

            let unresolved_ids: HashSet<&str> = shadow_report
                .scores
                .iter()
                .filter(|s| s.belief_state.as_deref() == Some("unresolved"))
                .map(|s| s.record_id.as_str())
                .collect();
            let baseline_unresolved = baseline_ids
                .iter()
                .take(effective_k)
                .filter(|id| unresolved_ids.contains(id.as_str()))
                .count();
            let limited_unresolved = limited_ids
                .iter()
                .take(effective_k)
                .filter(|id| unresolved_ids.contains(id.as_str()))
                .count();
            let contradiction_delta = limited_unresolved as i32 - baseline_unresolved as i32;

            let label = compute_quality_label(
                overlap,
                rerank_report.was_applied,
                rerank_report.records_moved,
                contradiction_delta,
                rerank_report.belief_coverage,
            );

        QueryMetrics {
            query: query.clone(),
            top_k_overlap: overlap,
            records_moved: rerank_report.records_moved,
                max_up_shift: rerank_report.max_up_shift,
                max_down_shift: rerank_report.max_down_shift,
                belief_coverage: rerank_report.belief_coverage,
                latency_delta_us: limited_latency as i64 - baseline_latency as i64,
                rerank_was_applied: rerank_report.was_applied,
                contradiction_delta,
                label,
            }
        })
        .collect()
}

fn compute_report(metrics: &[QueryMetrics]) -> RunReport {
    let n = metrics.len().max(1);
    let reranked = metrics.iter().filter(|m| m.rerank_was_applied).count();
    let better = metrics.iter().filter(|m| m.label == QualityLabel::Better).count();
    let same = metrics.iter().filter(|m| m.label == QualityLabel::Same).count();
    let worse = metrics.iter().filter(|m| m.label == QualityLabel::Worse).count();
    let avg_overlap = metrics.iter().map(|m| m.top_k_overlap).sum::<f32>() / n as f32;
    let avg_coverage = metrics.iter().map(|m| m.belief_coverage).sum::<f32>() / n as f32;
    let avg_latency = metrics.iter().map(|m| m.latency_delta_us).sum::<i64>() as f64 / n as f64;
    let max_up = metrics.iter().map(|m| m.max_up_shift).max().unwrap_or(0);
    let max_down = metrics.iter().map(|m| m.max_down_shift).max().unwrap_or(0);

    RunReport {
        reranked_pct: reranked as f32 / n as f32,
        better_pct: better as f32 / n as f32,
        same_pct: same as f32 / n as f32,
        worse_pct: worse as f32 / n as f32,
        avg_overlap,
        avg_coverage,
        avg_latency_delta_us: avg_latency,
        max_up_shift: max_up,
        max_down_shift: max_down,
    }
}

#[test]
fn belief_rerank_varied_monitor_10_runs() {
    let mut reports = Vec::new();

    for run in 0..RUNS {
        let seed = 0xC0FFEE_u64 + run as u64 * 17;
        let aura = open_temp_aura();
        build_varied_corpus(&aura, seed);
        for _ in 0..12 {
            aura.run_maintenance();
        }

        let queries = varied_queries(seed);
        let metrics = collect_metrics(&aura, &queries);
        let report = compute_report(&metrics);

        let contradiction_worsened = metrics.iter().filter(|m| m.contradiction_delta > 0).count();
        let contradiction_worsened_pct = contradiction_worsened as f32 / metrics.len().max(1) as f32;

        eprintln!(
            "run={} reranked={:.1}% better={:.1}% same={:.1}% worse={:.1}% overlap={:.3} coverage={:.1}% latency_delta={:.0}us shift=↑{} ↓{}",
            run + 1,
            report.reranked_pct * 100.0,
            report.better_pct * 100.0,
            report.same_pct * 100.0,
            report.worse_pct * 100.0,
            report.avg_overlap,
            report.avg_coverage * 100.0,
            report.avg_latency_delta_us,
            report.max_up_shift,
            report.max_down_shift
        );

        let worse_queries: Vec<&QueryMetrics> = metrics
            .iter()
            .filter(|m| m.label == QualityLabel::Worse)
            .collect();
        for m in &worse_queries {
            eprintln!(
                "  WORSE query='{}' overlap={:.3} moved={} contradiction_delta={} coverage={:.1}% reranked={}",
                m.query,
                m.top_k_overlap,
                m.records_moved,
                m.contradiction_delta,
                m.belief_coverage * 100.0,
                m.rerank_was_applied
            );
        }

        assert!(report.worse_pct <= ALERT_MAX_WORSE_PCT, "run {} worse rate too high", run + 1);
        assert!(report.avg_overlap >= ALERT_MIN_AVG_OVERLAP, "run {} overlap too low", run + 1);
        assert!(report.max_up_shift <= ALERT_MAX_POS_SHIFT, "run {} max up shift exceeded", run + 1);
        assert!(report.max_down_shift <= ALERT_MAX_POS_SHIFT, "run {} max down shift exceeded", run + 1);
        assert!(
            report.avg_latency_delta_us <= ALERT_MAX_AVG_LATENCY_US,
            "run {} avg latency delta too high",
            run + 1
        );
        assert!(
            contradiction_worsened_pct <= 0.05,
            "run {} contradiction worsened too often",
            run + 1
        );

        reports.push(report);
    }

    let n = reports.len() as f32;
    let avg_reranked = reports.iter().map(|r| r.reranked_pct).sum::<f32>() / n;
    let avg_better = reports.iter().map(|r| r.better_pct).sum::<f32>() / n;
    let avg_same = reports.iter().map(|r| r.same_pct).sum::<f32>() / n;
    let avg_worse = reports.iter().map(|r| r.worse_pct).sum::<f32>() / n;
    let avg_overlap = reports.iter().map(|r| r.avg_overlap).sum::<f32>() / n;
    let avg_coverage = reports.iter().map(|r| r.avg_coverage).sum::<f32>() / n;
    let avg_latency = reports.iter().map(|r| r.avg_latency_delta_us).sum::<f64>() / reports.len() as f64;
    let max_up = reports.iter().map(|r| r.max_up_shift).max().unwrap_or(0);
    let max_down = reports.iter().map(|r| r.max_down_shift).max().unwrap_or(0);

    eprintln!("\nVARIED SYNTHETIC SUMMARY ({} runs)", RUNS);
    eprintln!(
        "avg reranked={:.1}% better={:.1}% same={:.1}% worse={:.1}% overlap={:.3} coverage={:.1}% latency_delta={:.0}us max_shift=↑{} ↓{}",
        avg_reranked * 100.0,
        avg_better * 100.0,
        avg_same * 100.0,
        avg_worse * 100.0,
        avg_overlap,
        avg_coverage * 100.0,
        avg_latency,
        max_up,
        max_down
    );
}
