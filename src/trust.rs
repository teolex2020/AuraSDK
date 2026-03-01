//! Provenance stamping, trust scoring, and tag taxonomy.
//!
//! Rewritten from brain_tools.py trust/provenance logic.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ── Tag Taxonomy ──

/// User-configurable tag classification.
///
/// All maintenance phases reference `taxonomy` instead of hardcoded constants.
/// Ships with sensible defaults; user overrides via `brain.set_taxonomy()`.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct TagTaxonomy {
    /// Tags that mark records as permanent identity (never decayed, never archived).
    pub identity_tags: HashSet<String>,
    /// Tags that mark records as stable (low volatility, slow decay).
    pub stable_tags: HashSet<String>,
    /// Tags that mark records as volatile (high volatility, fast decay).
    pub volatile_tags: HashSet<String>,
    /// Tags that should NEVER be at IDENTITY level (transient/system data).
    pub non_identity_tags: HashSet<String>,
    /// Tags that should NEVER be consolidated (sensitive/unique records).
    pub consolidation_skip_tags: HashSet<String>,
    /// Tags that should NEVER be auto-archived (personal data).
    pub archive_protected_tags: HashSet<String>,
    /// Tags that trigger human approval before write (sensitive operations).
    pub sensitive_tags: HashSet<String>,
}

impl Default for TagTaxonomy {
    fn default() -> Self {
        Self {
            identity_tags: ["user-profile", "identity"]
                .iter().map(|s| s.to_string()).collect(),
            stable_tags: [
                "identity", "contact", "credential", "financial", "person",
            ].iter().map(|s| s.to_string()).collect(),
            volatile_tags: [
                "cache", "scheduled-task", "todo-item", "web-search-cache",
            ].iter().map(|s| s.to_string()).collect(),
            non_identity_tags: [
                "session-summary", "cache", "outcome", "plan", "reflection",
                "research-finding", "web-search-cache", "proactive-session",
                "action-plan", "session-reflection", "scheduled-task",
                "consolidated-meta", "research-project", "autonomous-outcome",
                "autonomous-goal",
            ].iter().map(|s| s.to_string()).collect(),
            consolidation_skip_tags: [
                "identity", "contact", "credential", "financial", "person",
                "user-profile", "session-summary", "scheduled-task",
                "health-metric", "extracted-fact", "todo-item",
            ].iter().map(|s| s.to_string()).collect(),
            archive_protected_tags: [
                "identity", "contact", "person", "health-metric",
                "extracted-fact", "relationship",
            ].iter().map(|s| s.to_string()).collect(),
            sensitive_tags: [
                "financial", "credential", "wallet",
            ].iter().map(|s| s.to_string()).collect(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl TagTaxonomy {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

// ── Trust Config ──

/// User-configurable trust scoring model.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Source → base trust score (0.0-1.0).
    pub source_trust: HashMap<String, f32>,
    /// Source → authority multiplier for recall ranking.
    pub source_authority: HashMap<String, f32>,
    /// Recency boost max (added to trust for fresh records).
    pub recency_boost_max: f32,
    /// Recency boost half-life in days.
    pub recency_half_life_days: f32,
}

impl Default for TrustConfig {
    fn default() -> Self {
        let mut source_trust = HashMap::new();
        source_trust.insert("user-confirmed".into(), 1.0);
        source_trust.insert("agent-interactive".into(), 0.7);
        source_trust.insert("system".into(), 0.6);
        source_trust.insert("agent".into(), 0.5);
        source_trust.insert("agent-autonomous".into(), 0.4);
        source_trust.insert("agent-worker".into(), 0.35);

        let mut source_authority = HashMap::new();
        source_authority.insert("user-telegram".into(), 1.2);
        source_authority.insert("user-desktop".into(), 1.2);
        source_authority.insert("user-voice".into(), 1.2);
        source_authority.insert("user-confirmed".into(), 1.2);
        source_authority.insert("agent-interactive".into(), 1.0);
        source_authority.insert("system".into(), 0.9);
        source_authority.insert("agent".into(), 0.85);
        source_authority.insert("agent-autonomous".into(), 0.75);
        source_authority.insert("agent-worker".into(), 0.7);
        source_authority.insert("agent-inference".into(), 0.65);

        Self {
            source_trust,
            source_authority,
            recency_boost_max: 0.2,
            recency_half_life_days: 7.0,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl TrustConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

// ── Provenance ──

/// Provenance metadata stamped on every stored record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    pub source: String,
    pub verified: bool,
    pub trust_score: f32,
    pub volatility: String,
    pub timestamp: String,
}

/// Infer volatility classification from record tags.
pub fn infer_volatility(tags: &[String], taxonomy: &TagTaxonomy) -> &'static str {
    let tag_set: HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();
    for t in &taxonomy.stable_tags {
        if tag_set.contains(t.as_str()) {
            return "stable";
        }
    }
    for t in &taxonomy.volatile_tags {
        if tag_set.contains(t.as_str()) {
            return "volatile";
        }
    }
    "moderate"
}

/// Get base provenance from a channel string.
pub fn get_provenance(channel: Option<&str>, trust_config: &TrustConfig) -> Provenance {
    let source = match channel {
        Some(ch) if ch == "telegram" || ch == "desktop" || ch == "voice" => {
            format!("user-{}", ch)
        }
        Some(ch) => ch.to_string(),
        None => "agent".to_string(),
    };

    let trust_score = *trust_config.source_trust
        .get(&source)
        .unwrap_or(&0.5);

    let verified = source.starts_with("user-");

    Provenance {
        source,
        verified,
        trust_score,
        volatility: "moderate".into(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

/// Stamp provenance into record metadata.
pub fn stamp_provenance(
    metadata: &mut HashMap<String, String>,
    channel: Option<&str>,
    tags: &[String],
    taxonomy: &TagTaxonomy,
    trust_config: &TrustConfig,
) {
    let prov = get_provenance(channel, trust_config);

    metadata.entry("source".into()).or_insert(prov.source);
    metadata.entry("verified".into()).or_insert(prov.verified.to_string());
    metadata.entry("trust_score".into()).or_insert(format!("{:.2}", prov.trust_score));
    metadata.entry("volatility".into()).or_insert_with(|| {
        infer_volatility(tags, taxonomy).to_string()
    });
    metadata.entry("timestamp".into()).or_insert(prov.timestamp);
}

/// Compute effective trust score at recall time.
///
/// `effective_trust = (base_trust + recency_boost) × authority_multiplier × source_type_factor`
///
/// `source_type_factor` encodes epistemological reliability:
/// recorded (direct user input) > retrieved (external source) > inferred (LLM reasoning) > generated (agent-created).
pub fn compute_effective_trust(
    metadata: &HashMap<String, String>,
    now_unix: f64,
    trust_config: &TrustConfig,
    source_type: &str,
) -> f32 {
    let trust = metadata
        .get("trust_score")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5);

    let source = metadata.get("source").map(|s| s.as_str()).unwrap_or("");

    // Source authority multiplier
    let authority = *trust_config.source_authority
        .get(source)
        .unwrap_or(&0.85);

    // Recency boost — fresh records get +max, decays over half_life days
    let timestamp_str = metadata.get("timestamp")
        .or_else(|| metadata.get("created_at"))
        .map(|s| s.as_str())
        .unwrap_or("");

    let ts = chrono::DateTime::parse_from_rfc3339(timestamp_str)
        .map(|dt| dt.timestamp() as f64)
        .unwrap_or(now_unix - 86400.0 * 14.0); // assume 14 days old if unknown

    let age_days = ((now_unix - ts) / 86400.0).max(0.0);
    let recency_boost = (trust_config.recency_boost_max
        * (1.0 - age_days as f32 / trust_config.recency_half_life_days))
        .max(0.0);

    // Source type factor — epistemological reliability
    let source_type_factor = match source_type {
        "recorded" => 1.0_f32,
        "retrieved" => 0.9,
        "inferred" => 0.85,
        "generated" => 0.8,
        _ => 0.9, // unknown defaults to retrieved-level
    };

    let effective = (trust + recency_boost) * authority * source_type_factor;
    effective.clamp(0.05, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_taxonomy() {
        let tax = TagTaxonomy::default();
        assert!(tax.identity_tags.contains("user-profile"));
        assert!(tax.stable_tags.contains("contact"));
        assert!(tax.volatile_tags.contains("cache"));
    }

    #[test]
    fn test_infer_volatility() {
        let tax = TagTaxonomy::default();
        let stable_tags = vec!["contact".into(), "test".into()];
        assert_eq!(infer_volatility(&stable_tags, &tax), "stable");

        let volatile_tags = vec!["cache".into()];
        assert_eq!(infer_volatility(&volatile_tags, &tax), "volatile");

        let moderate_tags = vec!["random".into()];
        assert_eq!(infer_volatility(&moderate_tags, &tax), "moderate");
    }

    #[test]
    fn test_compute_effective_trust() {
        let config = TrustConfig::default();
        let mut meta = HashMap::new();
        meta.insert("trust_score".into(), "0.8".into());
        meta.insert("source".into(), "user-telegram".into());
        meta.insert("timestamp".into(), chrono::Utc::now().to_rfc3339());

        let now = chrono::Utc::now().timestamp() as f64;
        let score = compute_effective_trust(&meta, now, &config, "recorded");
        // user-telegram authority = 1.2, trust = 0.8, recency = ~0.2, source_type = 1.0
        // effective = (0.8 + 0.2) * 1.2 * 1.0 = 1.2, clamped to 1.0
        assert!(score >= 0.9);
    }

    #[test]
    fn test_source_type_factor() {
        let config = TrustConfig::default();
        let mut meta = HashMap::new();
        meta.insert("trust_score".into(), "0.7".into());
        meta.insert("source".into(), "agent-interactive".into());
        // Use a timestamp in the past so recency_boost ≈ 0
        let old = chrono::Utc::now() - chrono::Duration::days(30);
        meta.insert("timestamp".into(), old.to_rfc3339());
        let now = chrono::Utc::now().timestamp() as f64;

        let recorded = compute_effective_trust(&meta, now, &config, "recorded");
        let retrieved = compute_effective_trust(&meta, now, &config, "retrieved");
        let inferred = compute_effective_trust(&meta, now, &config, "inferred");
        let generated = compute_effective_trust(&meta, now, &config, "generated");

        assert!(recorded > retrieved, "recorded should rank higher than retrieved");
        assert!(retrieved > inferred, "retrieved should rank higher than inferred");
        assert!(inferred > generated, "inferred should rank higher than generated");
    }

    #[test]
    fn test_stamp_provenance() {
        let tax = TagTaxonomy::default();
        let config = TrustConfig::default();
        let mut meta = HashMap::new();
        let tags = vec!["contact".into()];

        stamp_provenance(&mut meta, Some("telegram"), &tags, &tax, &config);

        assert_eq!(meta.get("source").unwrap(), "user-telegram");
        assert_eq!(meta.get("verified").unwrap(), "true");
        assert_eq!(meta.get("volatility").unwrap(), "stable");
    }
}
