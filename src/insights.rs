//! Pattern detection engine — phased zero-LLM detectors.
//!
//! Three detection phases inspired by DeepInnovator's knowledge synthesis:
//! - Phase 0: Record-level health (decay risks, promotion candidates)
//! - Phase 1: Relationship analysis (clusters, conflicts, co-activation, hubs, chains)
//! - Phase 2: Cross-domain discovery (serendipity, trending, knowledge gaps)

use crate::levels::Level;
use crate::record::Record;
use std::collections::{HashMap, HashSet};

/// Insight severity level.
#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detection phase — determines when a detector runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    /// Record-level health checks.
    RecordHealth,
    /// Relationship and structure analysis.
    Relationships,
    /// Cross-domain discovery and trend detection.
    CrossDomain,
}

/// A detected pattern or actionable insight.
#[derive(Debug, Clone)]
pub struct Insight {
    pub insight_type: String,
    pub severity: Severity,
    pub phase: Phase,
    pub record_ids: Vec<String>,
    pub description: String,
    /// Quantitative evidence supporting this insight.
    /// Keys depend on insight_type. Examples:
    /// - serendipity: {"tag_jaccard": "0.05", "connection_weight": "0.72"}
    /// - trending: {"velocity": "3.5", "activation_count": "12"}
    /// - cluster: {"cluster_size": "5", "tag": "rust"}
    pub evidence: HashMap<String, String>,
}

// ── Public API ──

/// Run all detectors across all phases and return insights.
pub fn detect_all(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut insights = Vec::new();
    insights.extend(detect_phase0(records));
    insights.extend(detect_phase1(records));
    insights.extend(detect_phase2(records));
    insights
}

/// Phase 0: Record-level health checks.
pub fn detect_phase0(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut insights = Vec::new();
    insights.extend(detect_decay_risks(records));
    insights.extend(detect_promotion_candidates(records));
    insights
}

/// Phase 1: Relationship and structure analysis.
pub fn detect_phase1(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut insights = Vec::new();
    insights.extend(detect_clusters(records));
    insights.extend(detect_conflicts(records));
    insights.extend(detect_stale_topics(records));
    insights.extend(detect_hot_topics(records));
    insights.extend(detect_coactivation_momentum(records));
    insights.extend(detect_graph_hubs(records));
    insights.extend(detect_causal_chains(records));
    insights
}

/// Phase 2: Cross-domain discovery.
pub fn detect_phase2(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut insights = Vec::new();
    insights.extend(detect_serendipity(records));
    insights.extend(detect_trending(records));
    insights.extend(detect_knowledge_gaps(records));
    insights.extend(detect_preference_patterns(records));
    insights.extend(detect_decision_conflicts(records));
    insights
}

// ── Phase 0: Record Health ──

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

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), at_risk.len().to_string());

    vec![Insight {
        insight_type: "decay_risks".into(),
        severity: Severity::High,
        phase: Phase::RecordHealth,
        description: format!(
            "{} records at risk of archival (strength < 0.15)",
            at_risk.len()
        ),
        record_ids: at_risk,
        evidence,
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

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), candidates.len().to_string());

    vec![Insight {
        insight_type: "promotion_candidates".into(),
        severity: Severity::Medium,
        phase: Phase::RecordHealth,
        description: format!("{} records eligible for promotion", candidates.len()),
        record_ids: candidates,
        evidence,
    }]
}

// ── Phase 1: Relationships ──

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
        .map(|(tag, ids)| {
            let mut evidence = HashMap::new();
            evidence.insert("tag".into(), tag.clone());
            evidence.insert("cluster_size".into(), ids.len().to_string());
            Insight {
                insight_type: "cluster".into(),
                severity: Severity::Low,
                phase: Phase::Relationships,
                description: format!("Cluster around tag '{}': {} records", tag, ids.len()),
                record_ids: ids,
                evidence,
            }
        })
        .collect()
}

/// 4. Conflicts — potential contradictory information.
///
/// Two signals:
/// a) Records with same tags but at different levels (WORKING vs IDENTITY)
/// b) Records explicitly marked as semantic_type="contradiction"
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

    // Signal A: Level-based conflicts
    for (tag, entries) in &tag_levels {
        let levels: HashSet<Level> = entries.iter().map(|(_, l)| *l).collect();
        if levels.contains(&Level::Working) && levels.contains(&Level::Identity) {
            let ids: Vec<String> = entries.iter().map(|(id, _)| id.clone()).collect();
            let mut evidence = HashMap::new();
            evidence.insert("tag".into(), tag.clone());
            evidence.insert("levels".into(), "Working+Identity".into());
            evidence.insert("signal".into(), "level_mismatch".into());
            insights.push(Insight {
                insight_type: "conflict".into(),
                severity: Severity::High,
                phase: Phase::Relationships,
                description: format!(
                    "Potential conflict on '{}': WORKING and IDENTITY records coexist",
                    tag
                ),
                record_ids: ids,
                evidence,
            });
        }
    }

    // Signal B: Explicitly marked contradictions
    let contradictions: Vec<&Record> = records
        .values()
        .filter(|r| r.semantic_type == "contradiction" && r.is_alive())
        .collect();

    if !contradictions.is_empty() {
        let ids: Vec<String> = contradictions.iter().map(|r| r.id.clone()).collect();
        let mut evidence = HashMap::new();
        evidence.insert("count".into(), ids.len().to_string());
        evidence.insert("signal".into(), "semantic_type".into());
        insights.push(Insight {
            insight_type: "explicit_contradiction".into(),
            severity: Severity::Critical,
            phase: Phase::Relationships,
            description: format!(
                "{} records explicitly marked as contradictions — require resolution",
                ids.len()
            ),
            record_ids: ids,
            evidence,
        });
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

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), stale.len().to_string());
    evidence.insert("threshold_days".into(), "14".into());

    vec![Insight {
        insight_type: "stale_topics".into(),
        severity: Severity::Low,
        phase: Phase::Relationships,
        description: format!("{} records not accessed in 14+ days", stale.len()),
        record_ids: stale,
        evidence,
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

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), hot.len().to_string());

    vec![Insight {
        insight_type: "hot_topics".into(),
        severity: Severity::Low,
        phase: Phase::Relationships,
        description: format!("{} hot records (frequently accessed recently)", hot.len()),
        record_ids: hot,
        evidence,
    }]
}

/// 7. Co-activation momentum — records often recalled together.
fn detect_coactivation_momentum(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut strong_pairs = Vec::new();

    for rec in records.values() {
        for (conn_id, weight) in &rec.connections {
            if *weight >= 0.5 && rec.id < *conn_id {
                let mut evidence = HashMap::new();
                evidence.insert("connection_weight".into(), format!("{:.2}", weight));
                strong_pairs.push(Insight {
                    insight_type: "coactivation_momentum".into(),
                    severity: Severity::Low,
                    phase: Phase::Relationships,
                    description: format!(
                        "Strong co-activation between {} and {} (weight={:.2})",
                        rec.id, conn_id, weight
                    ),
                    record_ids: vec![rec.id.clone(), conn_id.clone()],
                    evidence,
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

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), hubs.len().to_string());

    vec![Insight {
        insight_type: "graph_hubs".into(),
        severity: Severity::Medium,
        phase: Phase::Relationships,
        description: format!("{} hub records (10+ connections)", hubs.len()),
        record_ids: hubs,
        evidence,
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
            let child = records
                .values()
                .find(|r| r.caused_by_id.as_ref() == Some(&current_id));
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
        .map(|chain| {
            let mut evidence = HashMap::new();
            evidence.insert("chain_length".into(), chain.len().to_string());
            Insight {
                insight_type: "causal_chain".into(),
                severity: Severity::Low,
                phase: Phase::Relationships,
                description: format!("Causal chain of {} records", chain.len()),
                record_ids: chain,
                evidence,
            }
        })
        .collect()
}

// ── Phase 2: Cross-Domain Discovery ──

/// Minimum connection weight to consider for serendipity.
const SERENDIPITY_WEIGHT_THRESHOLD: f32 = 0.6;

/// Maximum tag Jaccard similarity for a pair to be "cross-domain".
const SERENDIPITY_JACCARD_CEILING: f32 = 0.2;

/// Minimum activation velocity to be considered trending.
const TRENDING_VELOCITY_THRESHOLD: f32 = 2.0;

/// 10. Serendipity — non-obvious cross-domain connections.
///
/// Detects pairs of records from different tag-domains that have strong
/// connection weights (built through co-recall, not tag overlap).
/// Filters:
/// - connection weight >= 0.6
/// - tag Jaccard < 0.2 (different domains)
/// - connection type is associative, reflective, or coactivation (not causal)
/// - both records are alive and important (importance >= 0.3)
pub fn detect_serendipity(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut insights = Vec::new();

    for rec in records.values() {
        if !rec.is_alive() || rec.importance() < 0.3 {
            continue;
        }

        for (conn_id, weight) in &rec.connections {
            if *weight < SERENDIPITY_WEIGHT_THRESHOLD {
                continue;
            }

            // Canonical pair ordering to avoid duplicates
            let pair = if rec.id < *conn_id {
                (rec.id.clone(), conn_id.clone())
            } else {
                (conn_id.clone(), rec.id.clone())
            };
            if seen.contains(&pair) {
                continue;
            }
            seen.insert(pair);

            let other = match records.get(conn_id) {
                Some(o) if o.is_alive() && o.importance() >= 0.3 => o,
                _ => continue,
            };

            // Skip causal connections — those are expected, not serendipitous
            if let Some(conn_type) = rec.connection_type(conn_id) {
                if conn_type == "causal" {
                    continue;
                }
            }

            // Namespace guard
            if rec.namespace != other.namespace {
                continue;
            }

            // Tag Jaccard similarity — low means different domains
            let jaccard = tag_jaccard(&rec.tags, &other.tags);
            if jaccard >= SERENDIPITY_JACCARD_CEILING {
                continue;
            }

            let mut evidence = HashMap::new();
            evidence.insert("connection_weight".into(), format!("{:.2}", weight));
            evidence.insert("tag_jaccard".into(), format!("{:.2}", jaccard));
            evidence.insert(
                "connection_type".into(),
                rec.connection_type(conn_id)
                    .unwrap_or("untyped")
                    .to_string(),
            );
            evidence.insert("importance_a".into(), format!("{:.2}", rec.importance()));
            evidence.insert("importance_b".into(), format!("{:.2}", other.importance()));

            // Build descriptive domain labels from tags
            let domain_a = rec
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "untagged".into());
            let domain_b = other
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "untagged".into());

            insights.push(Insight {
                insight_type: "serendipity".into(),
                severity: Severity::Medium,
                phase: Phase::CrossDomain,
                description: format!(
                    "Cross-domain connection: '{}' ({}) <-> '{}' ({}) weight={:.2}, jaccard={:.2}",
                    truncate(&rec.content, 40),
                    domain_a,
                    truncate(&other.content, 40),
                    domain_b,
                    weight,
                    jaccard,
                ),
                record_ids: vec![rec.id.clone(), conn_id.clone()],
                evidence,
            });
        }
    }

    insights
}

/// 11. Trending — records with accelerating activation velocity.
///
/// Uses the EMA-based activation_velocity field updated on each activate().
/// A velocity >= 2.0 means the record is being activated at least twice per day
/// on average (with exponential weighting toward recent activity).
pub fn detect_trending(records: &HashMap<String, Record>) -> Vec<Insight> {
    let mut trending: Vec<(String, f32)> = records
        .values()
        .filter(|r| {
            r.is_alive()
                && r.activation_velocity >= TRENDING_VELOCITY_THRESHOLD
                && r.days_since_activation() < 3.0
        })
        .map(|r| (r.id.clone(), r.activation_velocity))
        .collect();

    if trending.is_empty() {
        return vec![];
    }

    // Sort by velocity descending
    trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let ids: Vec<String> = trending.iter().map(|(id, _)| id.clone()).collect();
    let top_velocity = trending[0].1;

    let mut evidence = HashMap::new();
    evidence.insert("count".into(), trending.len().to_string());
    evidence.insert("top_velocity".into(), format!("{:.1}", top_velocity));
    evidence.insert(
        "threshold".into(),
        format!("{:.1}", TRENDING_VELOCITY_THRESHOLD),
    );

    vec![Insight {
        insight_type: "trending".into(),
        severity: Severity::Medium,
        phase: Phase::CrossDomain,
        description: format!(
            "{} records trending (velocity >= {:.0}): top velocity {:.1}",
            trending.len(),
            TRENDING_VELOCITY_THRESHOLD,
            top_velocity,
        ),
        record_ids: ids,
        evidence,
    }]
}

/// 12. Knowledge gaps — dense clusters with weak bridges between them.
///
/// Identifies tag-clusters that have few or no inter-cluster connections,
/// suggesting missing knowledge links that could be valuable.
pub fn detect_knowledge_gaps(records: &HashMap<String, Record>) -> Vec<Insight> {
    // Build tag -> record set
    let mut tag_groups: HashMap<String, HashSet<String>> = HashMap::new();
    for rec in records.values() {
        if !rec.is_alive() {
            continue;
        }
        for tag in &rec.tags {
            tag_groups
                .entry(tag.clone())
                .or_default()
                .insert(rec.id.clone());
        }
    }

    // Only consider substantial clusters (3+ records)
    let clusters: Vec<(String, HashSet<String>)> = tag_groups
        .into_iter()
        .filter(|(_, ids)| ids.len() >= 3)
        .collect();

    let mut insights = Vec::new();

    // Check pairs of clusters for weak bridges
    for i in 0..clusters.len() {
        for j in (i + 1)..clusters.len() {
            let (tag_a, ids_a) = &clusters[i];
            let (tag_b, ids_b) = &clusters[j];

            // Skip if clusters share many records (they're the same domain)
            let overlap: usize = ids_a.intersection(ids_b).count();
            if overlap > 0 {
                continue;
            }

            // Count cross-cluster connections
            let mut bridge_count = 0u32;
            let mut bridge_weight_sum = 0.0f32;
            for id_a in ids_a {
                if let Some(rec_a) = records.get(id_a) {
                    for id_b in ids_b {
                        if let Some(w) = rec_a.connections.get(id_b) {
                            bridge_count += 1;
                            bridge_weight_sum += w;
                        }
                    }
                }
            }

            // Gap = two substantial clusters with zero or very few bridges
            let max_possible = (ids_a.len() * ids_b.len()) as u32;
            if bridge_count <= 1 && max_possible >= 9 {
                let mut evidence = HashMap::new();
                evidence.insert("cluster_a_tag".into(), tag_a.clone());
                evidence.insert("cluster_b_tag".into(), tag_b.clone());
                evidence.insert("cluster_a_size".into(), ids_a.len().to_string());
                evidence.insert("cluster_b_size".into(), ids_b.len().to_string());
                evidence.insert("bridge_count".into(), bridge_count.to_string());
                evidence.insert(
                    "avg_bridge_weight".into(),
                    if bridge_count > 0 {
                        format!("{:.2}", bridge_weight_sum / bridge_count as f32)
                    } else {
                        "0.00".into()
                    },
                );

                insights.push(Insight {
                    insight_type: "knowledge_gap".into(),
                    severity: Severity::Low,
                    phase: Phase::CrossDomain,
                    description: format!(
                        "Knowledge gap between '{}' ({} records) and '{}' ({} records): {} bridge connections",
                        tag_a, ids_a.len(), tag_b, ids_b.len(), bridge_count,
                    ),
                    record_ids: Vec::new(), // No specific records — this is cluster-level
                    evidence,
                });
            }
        }
    }

    insights
}

/// 13. Preference patterns — clusters of preference records in the same domain.
///
/// Detects when multiple preferences accumulate around a topic, suggesting
/// a coherent style/taste pattern the agent should be aware of.
pub fn detect_preference_patterns(records: &HashMap<String, Record>) -> Vec<Insight> {
    // Group preference records by their first tag (domain)
    let mut domain_prefs: HashMap<String, Vec<String>> = HashMap::new();

    for rec in records.values() {
        if rec.semantic_type == "preference" && rec.is_alive() {
            let domain = rec
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "general".into());
            domain_prefs.entry(domain).or_default().push(rec.id.clone());
        }
    }

    domain_prefs
        .into_iter()
        .filter(|(_, ids)| ids.len() >= 2)
        .map(|(domain, ids)| {
            let mut evidence = HashMap::new();
            evidence.insert("domain".into(), domain.clone());
            evidence.insert("preference_count".into(), ids.len().to_string());
            Insight {
                insight_type: "preference_pattern".into(),
                severity: Severity::Medium,
                phase: Phase::CrossDomain,
                description: format!(
                    "Preference pattern in '{}': {} preference records form a style cluster",
                    domain,
                    ids.len(),
                ),
                record_ids: ids,
                evidence,
            }
        })
        .collect()
}

/// 14. Decision conflicts — decision records in the same domain that may contradict.
///
/// Detects when multiple decisions exist for similar topics, which may indicate
/// superseded or conflicting choices that need resolution.
pub fn detect_decision_conflicts(records: &HashMap<String, Record>) -> Vec<Insight> {
    let decisions: Vec<&Record> = records
        .values()
        .filter(|r| r.semantic_type == "decision" && r.is_alive())
        .collect();

    let mut insights = Vec::new();

    for i in 0..decisions.len() {
        for j in (i + 1)..decisions.len() {
            let a = decisions[i];
            let b = decisions[j];

            // Same namespace guard
            if a.namespace != b.namespace {
                continue;
            }

            // Check for shared tags (same domain)
            let jaccard = tag_jaccard(&a.tags, &b.tags);
            if jaccard < 0.3 {
                continue; // Different domains
            }

            // Both are decisions in the same domain — potential conflict
            let mut evidence = HashMap::new();
            evidence.insert("tag_overlap".into(), format!("{:.2}", jaccard));
            evidence.insert("strength_a".into(), format!("{:.2}", a.strength));
            evidence.insert("strength_b".into(), format!("{:.2}", b.strength));

            let shared_tags: Vec<&str> = a
                .tags
                .iter()
                .filter(|t| b.tags.contains(t))
                .map(|s| s.as_str())
                .collect();
            evidence.insert("shared_tags".into(), shared_tags.join(", "));

            insights.push(Insight {
                insight_type: "decision_conflict".into(),
                severity: Severity::High,
                phase: Phase::CrossDomain,
                description: format!(
                    "Potentially conflicting decisions on [{}]: '{}' vs '{}'",
                    shared_tags.join(", "),
                    truncate(&a.content, 50),
                    truncate(&b.content, 50),
                ),
                record_ids: vec![a.id.clone(), b.id.clone()],
                evidence,
            });
        }
    }

    insights
}

// ── Helpers ──

/// Tag Jaccard similarity between two tag vectors.
fn tag_jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0; // Both untagged = same domain
    }
    let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 1.0;
    }
    intersection as f32 / union as f32
}

/// Truncate a string to max chars, appending "..." if truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{}...", truncated)
    }
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
        assert_eq!(insights[0].phase, Phase::RecordHealth);
        assert!(insights[0].evidence.contains_key("count"));
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
        assert_eq!(insights[0].phase, Phase::RecordHealth);
    }

    #[test]
    fn test_serendipity_detection() {
        let mut records = HashMap::new();

        // Record A: "rust" domain
        let mut a = Record::new(
            "Rust is great for systems programming".into(),
            Level::Domain,
        );
        a.tags = vec!["rust".into(), "systems".into()];
        a.strength = 0.9;
        a.activation_count = 3;

        // Record B: "cooking" domain — totally different
        let mut b = Record::new("Italian pasta recipes are the best".into(), Level::Domain);
        b.tags = vec!["cooking".into(), "italian".into()];
        b.strength = 0.9;
        b.activation_count = 3;

        // Strong connection between them (built through co-recall)
        a.add_typed_connection(&b.id, 0.75, "coactivation");
        b.add_typed_connection(&a.id, 0.75, "coactivation");

        records.insert(a.id.clone(), a);
        records.insert(b.id.clone(), b);

        let insights = detect_serendipity(&records);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].insight_type, "serendipity");
        assert_eq!(insights[0].phase, Phase::CrossDomain);
        assert!(insights[0].evidence.contains_key("tag_jaccard"));
        assert!(insights[0].evidence.contains_key("connection_weight"));

        // Jaccard should be 0.0 (no shared tags)
        let jaccard: f32 = insights[0].evidence["tag_jaccard"].parse().unwrap();
        assert!(jaccard < 0.01);
    }

    #[test]
    fn test_serendipity_skips_causal() {
        let mut records = HashMap::new();

        let mut a = Record::new("Decision A".into(), Level::Domain);
        a.tags = vec!["arch".into()];
        a.strength = 0.9;
        a.activation_count = 3;

        let mut b = Record::new("Consequence B".into(), Level::Domain);
        b.tags = vec!["deploy".into()];
        b.strength = 0.9;
        b.activation_count = 3;

        // Causal connection — should NOT be serendipity
        a.add_typed_connection(&b.id, 0.8, "causal");
        b.add_typed_connection(&a.id, 0.8, "causal");

        records.insert(a.id.clone(), a);
        records.insert(b.id.clone(), b);

        let insights = detect_serendipity(&records);
        assert!(insights.is_empty());
    }

    #[test]
    fn test_serendipity_skips_same_domain() {
        let mut records = HashMap::new();

        let mut a = Record::new("Rust memory safety".into(), Level::Domain);
        a.tags = vec!["rust".into(), "safety".into()];
        a.strength = 0.9;
        a.activation_count = 3;

        let mut b = Record::new("Rust ownership model".into(), Level::Domain);
        b.tags = vec!["rust".into(), "ownership".into()];
        b.strength = 0.9;
        b.activation_count = 3;

        // Strong connection but same domain (shared "rust" tag)
        a.add_typed_connection(&b.id, 0.8, "associative");
        b.add_typed_connection(&a.id, 0.8, "associative");

        records.insert(a.id.clone(), a);
        records.insert(b.id.clone(), b);

        let insights = detect_serendipity(&records);
        // Jaccard = 1/3 = 0.33 > 0.2 threshold, so should be empty
        assert!(insights.is_empty());
    }

    #[test]
    fn test_trending_detection() {
        let mut records = HashMap::new();

        let mut r = Record::new("Hot topic".into(), Level::Working);
        r.activation_velocity = 3.5; // Well above threshold
        r.activation_count = 5;
        // Set last_activated to now (within 3 days)
        r.last_activated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        records.insert(r.id.clone(), r);

        let insights = detect_trending(&records);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].insight_type, "trending");
        assert_eq!(insights[0].phase, Phase::CrossDomain);
        assert!(insights[0].evidence.contains_key("top_velocity"));
    }

    #[test]
    fn test_trending_ignores_stale() {
        let mut records = HashMap::new();

        let mut r = Record::new("Was trending".into(), Level::Working);
        r.activation_velocity = 5.0; // High velocity
        r.activation_count = 10;
        // But last activated 5 days ago
        r.last_activated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            - 5.0 * 86400.0;
        records.insert(r.id.clone(), r);

        let insights = detect_trending(&records);
        assert!(insights.is_empty());
    }

    #[test]
    fn test_knowledge_gaps() {
        let mut records = HashMap::new();

        // Cluster A: 3 records tagged "rust"
        for i in 0..3 {
            let mut r = Record::new(format!("Rust topic {}", i), Level::Working);
            r.tags = vec!["rust".into()];
            records.insert(r.id.clone(), r);
        }

        // Cluster B: 3 records tagged "cooking"
        for i in 0..3 {
            let mut r = Record::new(format!("Cooking topic {}", i), Level::Working);
            r.tags = vec!["cooking".into()];
            records.insert(r.id.clone(), r);
        }

        // No connections between clusters
        let insights = detect_knowledge_gaps(&records);
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].insight_type, "knowledge_gap");
        assert_eq!(insights[0].phase, Phase::CrossDomain);
        assert_eq!(insights[0].evidence["bridge_count"], "0");
    }

    #[test]
    fn test_tag_jaccard() {
        assert!(
            (tag_jaccard(&["a".into(), "b".into()], &["b".into(), "c".into()],) - 1.0 / 3.0).abs()
                < 0.01
        );

        assert!((tag_jaccard(&["a".into()], &["b".into()],) - 0.0).abs() < 0.01);

        assert!(
            (tag_jaccard(&["a".into(), "b".into()], &["a".into(), "b".into()],) - 1.0).abs() < 0.01
        );
    }

    #[test]
    fn test_phase_separation() {
        let records = HashMap::new();

        let p0 = detect_phase0(&records);
        let p1 = detect_phase1(&records);
        let p2 = detect_phase2(&records);
        let all = detect_all(&records);

        assert_eq!(p0.len() + p1.len() + p2.len(), all.len());
    }

    #[test]
    fn test_detect_all_backward_compat() {
        // Ensure detect_all still returns all insight types
        let mut records = HashMap::new();
        let mut r = Record::new("at risk".into(), Level::Working);
        r.strength = 0.10;
        records.insert(r.id.clone(), r);

        let insights = detect_all(&records);
        assert!(!insights.is_empty());
        assert!(insights.iter().any(|i| i.insight_type == "decay_risks"));
    }
}
