//! Pattern detection engine — 9 zero-LLM detectors.
//!
//! Rewritten from aura-cognitive insights.py.

use std::collections::{HashMap, HashSet};
use crate::record::Record;
use crate::levels::Level;

/// Insight severity level.
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// A detected pattern or actionable insight.
#[derive(Debug, Clone)]
pub struct Insight {
    pub insight_type: String,
    pub severity: Severity,
    pub record_ids: Vec<String>,
    pub description: String,
}

/// Run all 9 detectors and return insights.
pub fn detect_all(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut insights = Vec::new();

    insights.extend(detect_decay_risks(records));
    insights.extend(detect_promotion_candidates(records));
    insights.extend(detect_clusters(records));
    insights.extend(detect_conflicts(records));
    insights.extend(detect_stale_topics(records));
    insights.extend(detect_hot_topics(records));
    insights.extend(detect_coactivation_momentum(records));
    insights.extend(detect_graph_hubs(records));
    insights.extend(detect_causal_chains(records));

    insights
}

/// 1. Decay risks — records about to die (strength < 0.15).
fn detect_decay_risks(records: &HashMap<String, Record>) -> Vec<Insight> {
    let at_risk: Vec<String> = records
        .values()
        .filter(|r| r.strength < 0.15 && r.strength >= 0.05)
        .map(|r| r.id.clone())
        .collect();

    if at_risk.is_empty() {
        return vec![];
    }

    vec![Insight {
        insight_type: "decay_risks".into(),
        severity: Severity::High,
        description: format!("{} records at risk of archival (strength < 0.15)", at_risk.len()),
        record_ids: at_risk,
    }]
}

/// 2. Promotion candidates — records ready for level-up.
fn detect_promotion_candidates(records: &HashMap<String, Record>) -> Vec<Insight> {
    let candidates: Vec<String> = records
        .values()
        .filter(|r| r.can_promote())
        .map(|r| r.id.clone())
        .collect();

    if candidates.is_empty() {
        return vec![];
    }

    vec![Insight {
        insight_type: "promotion_candidates".into(),
        severity: Severity::Medium,
        description: format!("{} records eligible for promotion", candidates.len()),
        record_ids: candidates,
    }]
}

/// 3. Clusters — groups of related records (shared tags).
fn detect_clusters(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut tag_groups: HashMap<String, Vec<String>> = HashMap::new();

    for rec in records.values() {
        for tag in &rec.tags {
            tag_groups
                .entry(tag.clone())
                .or_default()
                .push(rec.id.clone());
        }
    }

    tag_groups
        .into_iter()
        .filter(|(_, ids)| ids.len() >= 3)
        .map(|(tag, ids)| Insight {
            insight_type: "cluster".into(),
            severity: Severity::Low,
            description: format!("Cluster around tag '{}': {} records", tag, ids.len()),
            record_ids: ids,
        })
        .collect()
}

/// 4. Conflicts — potential contradictory information.
///
/// Heuristic: records with same tags but at different levels (WORKING vs IDENTITY)
/// may contain outdated vs current info.
fn detect_conflicts(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut tag_levels: HashMap<String, Vec<(String, Level)>> = HashMap::new();

    for rec in records.values() {
        for tag in &rec.tags {
            tag_levels
                .entry(tag.clone())
                .or_default()
                .push((rec.id.clone(), rec.level));
        }
    }

    let mut insights = Vec::new();
    for (tag, entries) in &tag_levels {
        let levels: HashSet<Level> = entries.iter().map(|(_, l)| *l).collect();
        if levels.contains(&Level::Working) && levels.contains(&Level::Identity) {
            let ids: Vec<String> = entries.iter().map(|(id, _)| id.clone()).collect();
            insights.push(Insight {
                insight_type: "conflict".into(),
                severity: Severity::High,
                description: format!(
                    "Potential conflict on '{}': WORKING and IDENTITY records coexist",
                    tag
                ),
                record_ids: ids,
            });
        }
    }

    insights
}

/// 5. Stale topics — not accessed in 14+ days.
fn detect_stale_topics(records: &HashMap<String, Record>) -> Vec<Insight> {
    let stale: Vec<String> = records
        .values()
        .filter(|r| r.days_since_activation() >= 14.0 && r.is_alive())
        .map(|r| r.id.clone())
        .collect();

    if stale.is_empty() {
        return vec![];
    }

    vec![Insight {
        insight_type: "stale_topics".into(),
        severity: Severity::Low,
        description: format!("{} records not accessed in 14+ days", stale.len()),
        record_ids: stale,
    }]
}

/// 6. Hot topics — frequently accessed recently (activated 3+ times, last access < 24h).
fn detect_hot_topics(records: &HashMap<String, Record>) -> Vec<Insight> {
    let hot: Vec<String> = records
        .values()
        .filter(|r| r.activation_count >= 3 && r.days_since_activation() < 1.0)
        .map(|r| r.id.clone())
        .collect();

    if hot.is_empty() {
        return vec![];
    }

    vec![Insight {
        insight_type: "hot_topics".into(),
        severity: Severity::Low,
        description: format!("{} hot records (frequently accessed recently)", hot.len()),
        record_ids: hot,
    }]
}

/// 7. Co-activation momentum — records often recalled together.
fn detect_coactivation_momentum(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut strong_pairs = Vec::new();

    for rec in records.values() {
        for (conn_id, weight) in &rec.connections {
            if *weight >= 0.5 && rec.id < *conn_id {
                strong_pairs.push(Insight {
                    insight_type: "coactivation_momentum".into(),
                    severity: Severity::Low,
                    description: format!(
                        "Strong co-activation between {} and {} (weight={:.2})",
                        rec.id, conn_id, weight
                    ),
                    record_ids: vec![rec.id.clone(), conn_id.clone()],
                });
            }
        }
    }

    strong_pairs
}

/// 8. Graph hubs — highly connected nodes (10+ connections).
fn detect_graph_hubs(records: &HashMap<String, Record>) -> Vec<Insight> {
    let hubs: Vec<String> = records
        .values()
        .filter(|r| r.connections.len() >= 10)
        .map(|r| r.id.clone())
        .collect();

    if hubs.is_empty() {
        return vec![];
    }

    vec![Insight {
        insight_type: "graph_hubs".into(),
        severity: Severity::Medium,
        description: format!("{} hub records (10+ connections)", hubs.len()),
        record_ids: hubs,
    }]
}

/// 9. Causal chains — records linked via caused_by_id.
fn detect_causal_chains(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut chains: Vec<Vec<String>> = Vec::new();

    // Find chain roots (records that are caused_by targets but have no caused_by themselves)
    let has_parent: HashSet<String> = records
        .values()
        .filter_map(|r| r.caused_by_id.clone())
        .collect();

    let roots: Vec<&Record> = records
        .values()
        .filter(|r| has_parent.contains(&r.id) && r.caused_by_id.is_none())
        .collect();

    for root in roots {
        let mut chain = vec![root.id.clone()];
        // Follow children
        let mut current_id = root.id.clone();
        for _ in 0..10 {
            let child = records.values().find(|r| r.caused_by_id.as_ref() == Some(&current_id));
            match child {
                Some(c) => {
                    chain.push(c.id.clone());
                    current_id = c.id.clone();
                }
                None => break,
            }
        }

        if chain.len() >= 2 {
            chains.push(chain);
        }
    }

    chains
        .into_iter()
        .map(|chain| Insight {
            insight_type: "causal_chain".into(),
            severity: Severity::Low,
            description: format!("Causal chain of {} records", chain.len()),
            record_ids: chain,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_risks() {
        let mut records = HashMap::new();
        let mut r = Record::new("at risk".into(), Level::Working);
        r.strength = 0.10;
        records.insert(r.id.clone(), r);

        let insights = detect_decay_risks(&records);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].insight_type, "decay_risks");
    }

    #[test]
    fn test_promotion_candidates() {
        let mut records = HashMap::new();
        let mut r = Record::new("promoted".into(), Level::Working);
        r.activation_count = 5;
        r.strength = 0.8;
        records.insert(r.id.clone(), r);

        let insights = detect_promotion_candidates(&records);
        assert_eq!(insights.len(), 1);
    }
}
