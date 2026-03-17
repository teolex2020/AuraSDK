//! Policy Hint Layer — advisory action hints over causal, concept, and belief layers.
//!
//! Fifth (top) tier of the cognitive hierarchy:
//!   Record → Belief → Concept → Causal Pattern → **Policy**
//!
//! Phase 1 constraints (read-only advisory):
//!   - Does NOT influence recall ranking, record merge, or agent behavior
//!   - Does NOT execute actions — only emits hints
//!   - Full rebuild each maintenance cycle (no persistent trust)
//!   - Every hint traces back to source causal/concept/belief/record IDs
//!   - Namespace barrier: no cross-namespace hints
//!   - Deterministic templates only (no free-form generation)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::belief::{BeliefEngine, BeliefState};
use crate::causal::{
    meets_counterevidence_gate, meets_counterfactual_gate, meets_evidence_gate, CausalEngine,
    CausalEvidenceMode, CausalState,
};
use crate::concept::{ConceptEngine, ConceptState};
use crate::record::Record;

// ── Constants ──

/// Minimum causal_strength to consider a pattern as policy seed.
const MIN_CAUSAL_STRENGTH_FOR_SEED: f32 = 0.65;
/// Minimum supporting observations before a causal pattern can seed policy hints.
const MIN_CAUSAL_SUPPORT_FOR_SEED: usize = 2;

/// Scoring weights for policy_strength.
const W_CAUSAL: f32 = 0.35;
const W_CONFIDENCE: f32 = 0.25;
const W_UTILITY: f32 = 0.20;
const W_STABILITY: f32 = 0.20;

/// State thresholds.
const STABLE_THRESHOLD: f32 = 0.75;
const CANDIDATE_THRESHOLD: f32 = 0.50;

/// Negative outcome keywords (tag or semantic_type driven).
const NEGATIVE_KEYWORDS: &[&str] = &[
    "error",
    "failure",
    "fail",
    "crash",
    "bug",
    "incident",
    "rollback",
    "revert",
    "risk",
    "vulnerability",
    "downtime",
    "outage",
    "regression",
    "contradiction",
    "conflict",
];

/// Positive outcome keywords.
const POSITIVE_KEYWORDS: &[&str] = &[
    "success",
    "improvement",
    "improve",
    "faster",
    "reliable",
    "stable",
    "healthy",
    "secure",
    "optimized",
    "resolved",
    "fixed",
    "deployed",
    "completed",
    "approved",
];

// ── PolicyActionKind ──

/// The type of advisory action suggested by a policy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyActionKind {
    /// Strong positive pattern — prefer this approach.
    Prefer,
    /// Moderate positive pattern — recommended but not strong.
    Recommend,
    /// Moderate negative pattern — verify before proceeding.
    VerifyFirst,
    /// Strong negative pattern — avoid this approach.
    Avoid,
    /// Strong pattern with insufficient data to classify polarity.
    Warn,
}

// ── PolicyState ──

/// Lifecycle state of a policy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyState {
    /// Meets threshold, not yet confirmed stable.
    Candidate,
    /// High confidence advisory hint.
    Stable,
    /// Suppressed by competing/conflicting hint.
    Suppressed,
    /// Below threshold — discarded.
    Rejected,
}

impl Default for PolicyState {
    fn default() -> Self {
        Self::Candidate
    }
}

// ── PolicyHint ──

/// A discovered advisory policy hint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyHint {
    /// Unique identifier.
    pub id: String,
    /// Stable identity key.
    pub key: String,
    /// Namespace this hint belongs to.
    pub namespace: String,
    /// Domain/topic of the hint (from tags or concept).
    pub domain: String,

    /// Type of advisory action.
    pub action_kind: PolicyActionKind,
    /// Deterministic recommendation text.
    pub recommendation: String,

    // ── Provenance ──
    /// Causal pattern IDs that triggered this hint.
    pub trigger_causal_ids: Vec<String>,
    /// Concept IDs that support this hint.
    pub trigger_concept_ids: Vec<String>,
    /// Belief IDs that support this hint.
    pub trigger_belief_ids: Vec<String>,
    /// Record IDs (transitive provenance — causes + effects).
    pub supporting_record_ids: Vec<String>,
    /// Cause-side record IDs only (used for suppression scope).
    pub cause_record_ids: Vec<String>,

    // ── Scoring ──
    /// Aggregated confidence from causal + concept + belief.
    pub confidence: f32,
    /// How useful/impactful the pattern is (from outcome strength).
    pub utility_score: f32,
    /// Risk signal (negative outcome weight * causal strength).
    pub risk_score: f32,
    /// Composite policy strength.
    pub policy_strength: f32,

    /// Current lifecycle state.
    pub state: PolicyState,
    /// Timestamp of last rebuild.
    pub last_updated: f64,
}

// ── PolicyReport ──

/// Per-cycle report returned by PolicyEngine::discover().
#[derive(Debug, Clone, Default)]
pub struct PolicyReport {
    /// Number of causal seeds considered.
    pub seeds_found: usize,
    /// Total policy hints after generation.
    pub hints_found: usize,
    /// Hints that reached Stable state.
    pub stable_hints: usize,
    /// Hints suppressed by conflict.
    pub suppressed_hints: usize,
    /// Hints that were Rejected.
    pub rejected_hints: usize,
    /// Average policy_strength across all hints.
    pub avg_policy_strength: f32,
}

// ── Internal: outcome polarity ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polarity {
    Positive,
    Negative,
    Neutral,
}

// ── PolicyEngine ──

/// Recall reranking mode for policy-hint-weighted influence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PolicyRerankMode {
    /// No policy influence on recall ranking. Default.
    #[default]
    Off = 0,
    /// Limited influence: apply bounded reranking (capped score delta + positional shift limit).
    Limited = 1,
}

impl PolicyRerankMode {
    /// Convert from u8 (for atomic storage). Invalid values → Off.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Limited,
            _ => Self::Off,
        }
    }
}

/// Policy hint discovery engine. Full rebuild each maintenance cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEngine {
    /// Discovered hints keyed by hint ID.
    pub hints: HashMap<String, PolicyHint>,
    /// Key → hint ID index for deduplication.
    pub key_index: HashMap<String, String>,
}

impl PolicyEngine {
    /// Create a fresh empty engine. Called on startup — never loaded from disk.
    pub fn new() -> Self {
        Self {
            hints: HashMap::new(),
            key_index: HashMap::new(),
        }
    }

    /// Full rebuild: discover policy hints from causal patterns, concepts, and beliefs.
    ///
    /// Algorithm (4 phases):
    ///   A. Select policy-worthy causal seeds
    ///   B. Classify outcome polarity
    ///   C. Map to action kind
    ///   D. Emit advisory hints with scoring
    pub fn discover(
        &mut self,
        causal_engine: &CausalEngine,
        concept_engine: &ConceptEngine,
        belief_engine: &BeliefEngine,
        records: &HashMap<String, Record>,
    ) -> PolicyReport {
        // Full rebuild
        self.hints.clear();
        self.key_index.clear();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Phase A: Select causal seeds — require stable lower-layer provenance
        let seeds: Vec<&crate::causal::CausalPattern> = causal_engine
            .patterns
            .values()
            .filter(|p| {
                // In ExplicitTrusted mode any Candidate or Stable pattern with explicit
                // support is a valid seed — bypass the strength floor since the user
                // already asserted the causal link.
                let strength_ok = p.state == CausalState::Stable
                    || (p.state == CausalState::Candidate
                        && p.causal_strength >= MIN_CAUSAL_STRENGTH_FOR_SEED)
                    || (causal_engine.evidence_mode == CausalEvidenceMode::ExplicitTrusted
                        && p.explicit_support_count >= 1
                        && p.state != CausalState::Rejected);
                // In ExplicitTrusted mode a single explicit link bypasses the support count
                // requirement — the user explicitly declared the causal relationship.
                let support_ok = p.support_count >= MIN_CAUSAL_SUPPORT_FOR_SEED
                    || (causal_engine.evidence_mode == CausalEvidenceMode::ExplicitTrusted
                        && p.explicit_support_count >= 1);
                let evidence_ok =
                    if meets_evidence_gate(p, CausalEvidenceMode::StrictRepeatedWindows) {
                        true
                    } else if causal_engine.evidence_mode == CausalEvidenceMode::ExplicitTrusted
                        && meets_evidence_gate(p, CausalEvidenceMode::ExplicitTrusted)
                    {
                        // ExplicitTrusted: user-declared links already passed the causal
                        // gate — accept them as seeds without demanding strict repeated windows.
                        true
                    } else {
                        // Comparison-only recovery path for purely temporal clustered evidence.
                        // Do not allow it to piggyback on causes that already have explicit
                        // outcome mass, otherwise old confounder packs reopen through temporal
                        // side-paths around the explicit guards.
                        causal_engine.evidence_mode == CausalEvidenceMode::TemporalClusterRecovery
                            && meets_evidence_gate(p, CausalEvidenceMode::TemporalClusterRecovery)
                            && p.explicit_support_count == 0
                            && p.explicit_support_total_for_cause == 0
                            && p.counterevidence == 0
                            && p.positive_effect_signals > 0
                            && p.negative_effect_signals == 0
                    };
                let counterevidence_ok = meets_counterevidence_gate(p);
                let counterfactual_ok = meets_counterfactual_gate(p, causal_engine.evidence_mode);
                // Gate: at least one cause-side belief must be Resolved or Singleton,
                // OR in ExplicitTrusted mode the user-declared explicit link substitutes
                // for stable belief backing (useful when corpus is too small for beliefs).
                let has_stable_belief = p.cause_belief_ids.iter().any(|bid| {
                    belief_engine.beliefs.get(bid).is_some_and(|b| {
                        matches!(b.state, BeliefState::Resolved | BeliefState::Singleton)
                    })
                });
                let belief_gate_ok = has_stable_belief
                    || (causal_engine.evidence_mode == CausalEvidenceMode::ExplicitTrusted
                        && p.explicit_support_count >= 1);
                strength_ok
                    && support_ok
                    && evidence_ok
                    && counterevidence_ok
                    && counterfactual_ok
                    && belief_gate_ok
            })
            .collect();

        let seeds_found = seeds.len();
        if seeds.is_empty() {
            return PolicyReport::default();
        }

        // Build concept lookup: belief_id → concept_ids
        let belief_to_concepts = Self::build_belief_to_concepts(concept_engine);

        let mut hints_found = 0;
        let mut stable_hints = 0;
        let mut suppressed_hints = 0;
        let mut rejected_hints = 0;
        let mut strength_sum = 0.0f32;

        for pattern in &seeds {
            // Phase B: Classify outcome polarity from effect records
            let polarity = self.classify_polarity(pattern, records);
            let mixed_explicit_ambiguity =
                self.has_mixed_explicit_outcome_ambiguity(pattern, records);

            if mixed_explicit_ambiguity {
                continue;
            }

            // Phase C: Map to action kind
            let action_kind = Self::map_action_kind(polarity, pattern.causal_strength);

            // Phase D: Build the hint
            let hint = self.build_hint(
                pattern,
                polarity,
                action_kind,
                &belief_to_concepts,
                belief_engine,
                records,
                now,
            );

            let key = hint.key.clone();
            let id = hint.id.clone();
            let strength = hint.policy_strength;

            self.key_index.insert(key, id.clone());
            self.hints.insert(id, hint);

            hints_found += 1;
            strength_sum += strength;
        }

        // Phase E: Suppression — detect conflicting hints
        self.apply_suppression();

        // Classify states
        for hint in self.hints.values_mut() {
            if hint.state == PolicyState::Suppressed {
                suppressed_hints += 1;
                continue;
            }
            hint.state = if hint.policy_strength >= STABLE_THRESHOLD {
                PolicyState::Stable
            } else if hint.policy_strength >= CANDIDATE_THRESHOLD {
                PolicyState::Candidate
            } else {
                PolicyState::Rejected
            };
            match hint.state {
                PolicyState::Stable => stable_hints += 1,
                PolicyState::Rejected => rejected_hints += 1,
                _ => {}
            }
        }

        let avg_policy_strength = if hints_found > 0 {
            strength_sum / hints_found as f32
        } else {
            0.0
        };

        PolicyReport {
            seeds_found,
            hints_found,
            stable_hints,
            suppressed_hints,
            rejected_hints,
            avg_policy_strength,
        }
    }

    // ── Phase B: Polarity classification ──

    /// Classify the outcome polarity of a causal pattern by examining
    /// effect-side records' tags, semantic_type, and content keywords.
    fn classify_polarity(
        &self,
        pattern: &crate::causal::CausalPattern,
        records: &HashMap<String, Record>,
    ) -> Polarity {
        let (positive_signals, negative_signals) = self.polarity_signal_counts(pattern, records);

        if negative_signals > positive_signals && negative_signals >= 2 {
            Polarity::Negative
        } else if positive_signals > negative_signals && positive_signals >= 2 {
            Polarity::Positive
        } else {
            Polarity::Neutral
        }
    }

    fn has_mixed_explicit_outcome_ambiguity(
        &self,
        pattern: &crate::causal::CausalPattern,
        records: &HashMap<String, Record>,
    ) -> bool {
        if pattern.explicit_support_count < MIN_CAUSAL_SUPPORT_FOR_SEED
            || pattern.effect_record_signature_variants <= 1
        {
            return false;
        }

        let (positive_signals, negative_signals) = self.polarity_signal_counts(pattern, records);
        positive_signals >= 2 && negative_signals >= 2
    }

    fn polarity_signal_counts(
        &self,
        pattern: &crate::causal::CausalPattern,
        records: &HashMap<String, Record>,
    ) -> (usize, usize) {
        let mut positive_signals = 0;
        let mut negative_signals = 0;

        for eid in &pattern.effect_record_ids {
            if let Some(rec) = records.get(eid) {
                // Check semantic_type
                if rec.semantic_type == "contradiction" {
                    negative_signals += 2;
                }

                // Check tags
                for tag in &rec.tags {
                    let tag_lower = tag.to_lowercase();
                    if NEGATIVE_KEYWORDS.iter().any(|kw| tag_lower.contains(kw)) {
                        negative_signals += 1;
                    }
                    if POSITIVE_KEYWORDS.iter().any(|kw| tag_lower.contains(kw)) {
                        positive_signals += 1;
                    }
                }

                // Check content keywords (lightweight — no NLP)
                let content_lower = rec.content.to_lowercase();
                for kw in NEGATIVE_KEYWORDS {
                    if content_lower.contains(kw) {
                        negative_signals += 1;
                    }
                }
                for kw in POSITIVE_KEYWORDS {
                    if content_lower.contains(kw) {
                        positive_signals += 1;
                    }
                }
            }
        }

        (positive_signals, negative_signals)
    }

    // ── Phase C: Action kind mapping ──

    fn map_action_kind(polarity: Polarity, causal_strength: f32) -> PolicyActionKind {
        match polarity {
            Polarity::Negative if causal_strength >= 0.75 => PolicyActionKind::Avoid,
            Polarity::Negative => PolicyActionKind::VerifyFirst,
            Polarity::Positive if causal_strength >= 0.75 => PolicyActionKind::Prefer,
            Polarity::Positive => PolicyActionKind::Recommend,
            Polarity::Neutral => PolicyActionKind::Warn,
        }
    }

    // ── Phase D: Hint construction ──

    fn build_hint(
        &self,
        pattern: &crate::causal::CausalPattern,
        polarity: Polarity,
        action_kind: PolicyActionKind,
        belief_to_concepts: &HashMap<String, Vec<String>>,
        belief_engine: &BeliefEngine,
        records: &HashMap<String, Record>,
        now: f64,
    ) -> PolicyHint {
        // Build domain from tags of cause records
        let domain = self.extract_domain(pattern, records);

        // Stable key from namespace + action_kind + causal pattern key
        let key = format!(
            "{}:{}:{}",
            pattern.namespace,
            action_kind_str(action_kind),
            pattern.key
        );
        let id = deterministic_id(&key);

        // Collect concept IDs via belief → concept mapping
        let mut concept_ids = Vec::new();
        for bid in pattern
            .cause_belief_ids
            .iter()
            .chain(pattern.effect_belief_ids.iter())
        {
            if let Some(cids) = belief_to_concepts.get(bid) {
                for cid in cids {
                    if !concept_ids.contains(cid) {
                        concept_ids.push(cid.clone());
                    }
                }
            }
        }

        // Collect all belief IDs
        let mut belief_ids: Vec<String> = pattern.cause_belief_ids.clone();
        for bid in &pattern.effect_belief_ids {
            if !belief_ids.contains(bid) {
                belief_ids.push(bid.clone());
            }
        }

        // Collect all record IDs
        let mut record_ids: Vec<String> = pattern.cause_record_ids.clone();
        for rid in &pattern.effect_record_ids {
            if !record_ids.contains(rid) {
                record_ids.push(rid.clone());
            }
        }

        // Scoring
        let causal_strength = pattern.causal_strength;

        // Confidence: aggregate from beliefs; fall back to record-level confidence when the
        // pattern has no belief backing (orphan records in ExplicitTrusted mode).
        let belief_confidence = self.aggregate_belief_confidence(&belief_ids, belief_engine);
        let confidence = if belief_confidence > 0.0 {
            belief_confidence
        } else {
            // Fallback: average confidence of backing records (trust encoded in record.confidence)
            let vals: Vec<f32> = record_ids
                .iter()
                .filter_map(|rid| records.get(rid))
                .map(|r| r.confidence)
                .collect();
            if vals.is_empty() {
                0.50 // neutral default
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            }
        };

        // Utility: based on pattern support and outcome consistency
        let utility_score = (pattern.outcome_stability * pattern.temporal_consistency).min(1.0);

        // Risk: negative polarity amplifies risk
        let risk_score = match polarity {
            Polarity::Negative => causal_strength * 0.8,
            Polarity::Neutral => causal_strength * 0.3,
            Polarity::Positive => 0.0,
        };

        // Stability proxy: use causal temporal_consistency as a proxy for now
        let stability = pattern.temporal_consistency;

        let policy_strength = W_CAUSAL * causal_strength
            + W_CONFIDENCE * confidence
            + W_UTILITY * utility_score
            + W_STABILITY * stability;

        // Generate deterministic recommendation text
        let recommendation = Self::generate_recommendation(action_kind, &domain, pattern, records);

        PolicyHint {
            id,
            key,
            namespace: pattern.namespace.clone(),
            domain,
            action_kind,
            recommendation,
            trigger_causal_ids: vec![pattern.id.clone()],
            trigger_concept_ids: concept_ids,
            trigger_belief_ids: belief_ids,
            supporting_record_ids: record_ids,
            cause_record_ids: pattern.cause_record_ids.clone(),
            confidence,
            utility_score,
            risk_score,
            policy_strength,
            state: PolicyState::Candidate, // classified later
            last_updated: now,
        }
    }

    /// Extract domain string from the most common tags of cause records.
    fn extract_domain(
        &self,
        pattern: &crate::causal::CausalPattern,
        records: &HashMap<String, Record>,
    ) -> String {
        let mut tag_counts: HashMap<&str, usize> = HashMap::new();
        for rid in &pattern.cause_record_ids {
            if let Some(rec) = records.get(rid) {
                for tag in &rec.tags {
                    *tag_counts.entry(tag.as_str()).or_default() += 1;
                }
            }
        }
        let mut tags: Vec<(&&str, &usize)> = tag_counts.iter().collect();
        tags.sort_by(|a, b| b.1.cmp(a.1));
        tags.iter()
            .take(2)
            .map(|(t, _)| **t)
            .collect::<Vec<_>>()
            .join("/")
    }

    /// Aggregate confidence from resolved beliefs.
    fn aggregate_belief_confidence(
        &self,
        belief_ids: &[String],
        belief_engine: &BeliefEngine,
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0;
        for bid in belief_ids {
            if let Some(belief) = belief_engine.beliefs.get(bid) {
                match belief.state {
                    BeliefState::Resolved | BeliefState::Singleton => {
                        sum += belief.confidence;
                        count += 1;
                    }
                    _ => {}
                }
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Generate deterministic recommendation text from template.
    fn generate_recommendation(
        action_kind: PolicyActionKind,
        domain: &str,
        pattern: &crate::causal::CausalPattern,
        records: &HashMap<String, Record>,
    ) -> String {
        // Get a brief cause summary from first cause record
        let cause_summary = pattern
            .cause_record_ids
            .first()
            .and_then(|rid| records.get(rid))
            .map(|r| truncate(&r.content, 80))
            .unwrap_or_else(|| "this action".to_string());

        match action_kind {
            PolicyActionKind::Avoid => {
                format!(
                    "Avoid: '{}' in domain [{}] has been associated with negative outcomes.",
                    cause_summary, domain
                )
            }
            PolicyActionKind::VerifyFirst => {
                format!("Verify first: '{}' in domain [{}] has shown risk signals — check before proceeding.", cause_summary, domain)
            }
            PolicyActionKind::Prefer => {
                format!(
                    "Prefer: '{}' in domain [{}] has consistently led to positive outcomes.",
                    cause_summary, domain
                )
            }
            PolicyActionKind::Recommend => {
                format!(
                    "Recommend: '{}' in domain [{}] has shown positive signals.",
                    cause_summary, domain
                )
            }
            PolicyActionKind::Warn => {
                format!("Warning: '{}' in domain [{}] has a strong causal pattern but unclear polarity.", cause_summary, domain)
            }
        }
    }

    // ── Phase E: Suppression ──

    /// Detect conflicting hints in the same namespace+domain and suppress the weaker one.
    /// Conflict: one hint says Prefer/Recommend, another says Avoid/VerifyFirst for
    /// overlapping cause records.
    fn apply_suppression(&mut self) {
        let hint_ids: Vec<String> = self.hints.keys().cloned().collect();
        let mut to_suppress: Vec<String> = Vec::new();

        for i in 0..hint_ids.len() {
            for j in (i + 1)..hint_ids.len() {
                let a = match self.hints.get(&hint_ids[i]) {
                    Some(h) => h,
                    None => continue,
                };
                let b = match self.hints.get(&hint_ids[j]) {
                    Some(h) => h,
                    None => continue,
                };

                // Same namespace + overlapping domain
                if a.namespace != b.namespace || a.domain != b.domain {
                    continue;
                }

                // Check if they conflict: one positive, one negative
                let a_positive = matches!(
                    a.action_kind,
                    PolicyActionKind::Prefer | PolicyActionKind::Recommend
                );
                let b_positive = matches!(
                    b.action_kind,
                    PolicyActionKind::Prefer | PolicyActionKind::Recommend
                );

                if a_positive == b_positive {
                    continue; // same direction — no conflict
                }

                // Check overlapping cause-side records only (not effects)
                let a_causes: std::collections::HashSet<&str> =
                    a.cause_record_ids.iter().map(|s| s.as_str()).collect();
                let b_causes: std::collections::HashSet<&str> =
                    b.cause_record_ids.iter().map(|s| s.as_str()).collect();
                let overlap = a_causes.intersection(&b_causes).count();

                if overlap == 0 {
                    continue; // no overlap — not a real conflict
                }

                // Suppress the weaker one
                if a.policy_strength < b.policy_strength {
                    to_suppress.push(hint_ids[i].clone());
                } else {
                    to_suppress.push(hint_ids[j].clone());
                }
            }
        }

        for id in to_suppress {
            if let Some(hint) = self.hints.get_mut(&id) {
                hint.state = PolicyState::Suppressed;
            }
        }
    }

    /// Build a reverse map: belief_id → concept_ids that include this belief.
    fn build_belief_to_concepts(concept_engine: &ConceptEngine) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for (cid, concept) in &concept_engine.concepts {
            if concept.state != ConceptState::Stable {
                continue; // only stable concepts
            }
            for bid in &concept.belief_ids {
                map.entry(bid.clone()).or_default().push(cid.clone());
            }
        }
        map
    }
}

/// Generate a deterministic policy hint ID from its stable key.
fn deterministic_id(key: &str) -> String {
    let hash = xxhash_rust::xxh3::xxh3_64(key.as_bytes());
    format!("p-{:012x}", hash)
}

// ── Helpers ──

fn action_kind_str(kind: PolicyActionKind) -> &'static str {
    match kind {
        PolicyActionKind::Prefer => "prefer",
        PolicyActionKind::Recommend => "recommend",
        PolicyActionKind::VerifyFirst => "verify",
        PolicyActionKind::Avoid => "avoid",
        PolicyActionKind::Warn => "warn",
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

// ── PolicyStore ──

/// Persistent store for policy hints. Write-only cache for debugging/inspection.
/// Never loaded on startup.
#[derive(Debug)]
pub struct PolicyStore {
    path: std::path::PathBuf,
}

impl PolicyStore {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Save current engine state to policies.cog (best-effort).
    pub fn save(&self, engine: &PolicyEngine) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.path)?;
        let file_path = self.path.join("policies.cog");
        let data = serde_json::to_vec(engine)?;
        std::fs::write(&file_path, data)?;
        Ok(())
    }

    /// Load from disk. Inspection-only utility — NOT called on startup.
    pub fn load(&self) -> anyhow::Result<PolicyEngine> {
        let file_path = self.path.join("policies.cog");
        if !file_path.exists() {
            return Ok(PolicyEngine::new());
        }
        let data = std::fs::read(&file_path)?;
        let engine: PolicyEngine = serde_json::from_slice(&data)?;
        Ok(engine)
    }
}

// ════════════════════════════════════════════════════════════
// Surfaced Policy Output
// ════════════════════════════════════════════════════════════

/// Minimum policy_strength for a Candidate to be surfaced.
const STRONG_CANDIDATE_THRESHOLD: f32 = 0.70;

/// Minimum confidence for surfacing.
const MIN_SURFACE_CONFIDENCE: f32 = 0.55;

/// Maximum total surfaced hints.
const MAX_SURFACED_HINTS: usize = 10;

/// Maximum surfaced hints per domain.
const MAX_SURFACED_PER_DOMAIN: usize = 3;

/// A filtered, stable, user-facing advisory hint.
/// This is the external contract — decoupled from internal PolicyHint.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacedPolicyHint {
    /// Hint identifier.
    pub id: String,
    /// Lifecycle state ("stable" or "candidate").
    pub state: String,
    /// Advisory action kind ("prefer", "recommend", "verify_first", "avoid", "warn").
    pub action_kind: String,

    /// Namespace this hint belongs to.
    pub namespace: String,
    /// Domain/topic of the hint.
    pub domain: String,

    /// Human-readable recommendation text.
    pub recommendation: String,

    /// Composite policy strength score.
    pub policy_strength: f32,
    /// Aggregated confidence from beliefs.
    pub confidence: f32,
    /// Risk signal (higher = more negative outcome weight).
    pub risk_score: f32,

    // ── Provenance ──
    /// Causal pattern IDs that triggered this hint.
    pub trigger_causal_ids: Vec<String>,
    /// Concept IDs that support this hint.
    pub trigger_concept_ids: Vec<String>,
    /// Belief IDs that support this hint.
    pub trigger_belief_ids: Vec<String>,
    /// Record IDs (transitive provenance).
    pub supporting_record_ids: Vec<String>,
}

/// Surface filtering and sorting for policy hints.
/// Takes the current engine state and returns a bounded, sorted, provenance-checked
/// list of advisory hints suitable for external consumption.
pub fn surface_policy_hints(
    engine: &PolicyEngine,
    limit: Option<usize>,
) -> Vec<SurfacedPolicyHint> {
    surface_policy_hints_filtered(engine, limit, None)
}

/// Surface with optional namespace filter.
pub fn surface_policy_hints_filtered(
    engine: &PolicyEngine,
    limit: Option<usize>,
    namespace: Option<&str>,
) -> Vec<SurfacedPolicyHint> {
    let max = limit.unwrap_or(MAX_SURFACED_HINTS).min(MAX_SURFACED_HINTS);

    // Phase A+B: filter eligible hints
    let mut eligible: Vec<&PolicyHint> = engine
        .hints
        .values()
        .filter(|h| {
            // Namespace filter
            if let Some(ns) = namespace {
                if h.namespace != ns {
                    return false;
                }
            }

            // Must have provenance
            if h.trigger_causal_ids.is_empty() || h.supporting_record_ids.is_empty() {
                return false;
            }

            // Must have non-empty domain and recommendation
            if h.domain.is_empty() || h.recommendation.is_empty() {
                return false;
            }

            // State gate
            match h.state {
                PolicyState::Stable => true,
                PolicyState::Candidate => {
                    h.policy_strength >= STRONG_CANDIDATE_THRESHOLD
                        && h.confidence >= MIN_SURFACE_CONFIDENCE
                }
                PolicyState::Suppressed | PolicyState::Rejected => false,
            }
        })
        .collect();

    // Phase C: sort deterministically
    // Higher policy_strength > higher confidence > higher risk_score > stable over candidate > key tiebreak
    eligible.sort_by(|a, b| {
        b.policy_strength
            .partial_cmp(&a.policy_strength)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                b.risk_score
                    .partial_cmp(&a.risk_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then_with(|| {
                let a_stable = matches!(a.state, PolicyState::Stable);
                let b_stable = matches!(b.state, PolicyState::Stable);
                b_stable.cmp(&a_stable)
            })
            .then(a.key.cmp(&b.key))
    });

    // Phase D: per-domain cap + global limit + dedup by key
    let mut result = Vec::new();
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    let mut seen_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_recommendations: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for hint in &eligible {
        if result.len() >= max {
            break;
        }

        // Dedupe by key
        if !seen_keys.insert(hint.key.clone()) {
            continue;
        }

        // Dedupe identical recommendation text
        if !seen_recommendations.insert(hint.recommendation.clone()) {
            continue;
        }

        // Per-domain cap
        let count = domain_counts.entry(hint.domain.clone()).or_default();
        if *count >= MAX_SURFACED_PER_DOMAIN {
            continue;
        }
        *count += 1;

        // Phase E: map to surfaced type
        result.push(SurfacedPolicyHint {
            id: hint.id.clone(),
            state: match hint.state {
                PolicyState::Stable => "stable".to_string(),
                PolicyState::Candidate => "candidate".to_string(),
                _ => unreachable!(), // filtered above
            },
            action_kind: action_kind_str(hint.action_kind).to_string(),
            namespace: hint.namespace.clone(),
            domain: hint.domain.clone(),
            recommendation: truncate(&hint.recommendation, 200),
            policy_strength: hint.policy_strength,
            confidence: hint.confidence,
            risk_score: hint.risk_score,
            trigger_causal_ids: hint.trigger_causal_ids.clone(),
            trigger_concept_ids: hint.trigger_concept_ids.clone(),
            trigger_belief_ids: hint.trigger_belief_ids.clone(),
            supporting_record_ids: hint.supporting_record_ids.clone(),
        });
    }

    result
}

// ════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefEngine;
    use crate::causal::{CausalEngine, CausalPattern, CausalState};
    use crate::concept::ConceptEngine;
    use crate::levels::Level;

    fn make_record(id: &str, content: &str, ns: &str, tags: &[&str], semantic: &str) -> Record {
        let mut rec = Record::new(content.to_string(), Level::Domain);
        rec.id = id.to_string();
        rec.namespace = ns.to_string();
        rec.tags = tags.iter().map(|t| t.to_string()).collect();
        rec.semantic_type = semantic.to_string();
        rec
    }

    /// Insert a Singleton belief into the engine, return its ID.
    fn inject_singleton_belief(engine: &mut BeliefEngine, bid: &str) {
        use crate::belief::Belief;
        let mut b = Belief::new(format!("test_key_{}", bid));
        b.id = bid.to_string();
        b.state = BeliefState::Singleton;
        b.confidence = 0.75;
        engine.beliefs.insert(bid.to_string(), b);
    }

    fn make_causal_pattern(
        id: &str,
        ns: &str,
        cause_rids: &[&str],
        effect_rids: &[&str],
        strength: f32,
        state: CausalState,
    ) -> CausalPattern {
        make_causal_pattern_with_beliefs(
            id,
            ns,
            cause_rids,
            effect_rids,
            &["b_cause"],
            &[],
            strength,
            state,
        )
    }

    fn make_causal_pattern_with_beliefs(
        id: &str,
        ns: &str,
        cause_rids: &[&str],
        effect_rids: &[&str],
        cause_bids: &[&str],
        effect_bids: &[&str],
        strength: f32,
        state: CausalState,
    ) -> CausalPattern {
        make_causal_pattern_with_evidence(
            id,
            ns,
            cause_rids,
            effect_rids,
            cause_bids,
            effect_bids,
            strength,
            state,
            3,
            1,
            2,
        )
    }

    fn make_causal_pattern_with_evidence(
        id: &str,
        ns: &str,
        cause_rids: &[&str],
        effect_rids: &[&str],
        cause_bids: &[&str],
        effect_bids: &[&str],
        strength: f32,
        state: CausalState,
        support_count: usize,
        explicit_support_count: usize,
        unique_temporal_windows: usize,
    ) -> CausalPattern {
        CausalPattern {
            id: id.to_string(),
            key: format!("{}:test_key:{}", ns, id),
            namespace: ns.to_string(),
            cause_belief_ids: cause_bids.iter().map(|s| s.to_string()).collect(),
            effect_belief_ids: effect_bids.iter().map(|s| s.to_string()).collect(),
            cause_record_ids: cause_rids.iter().map(|s| s.to_string()).collect(),
            effect_record_ids: effect_rids.iter().map(|s| s.to_string()).collect(),
            support_count,
            explicit_support_count,
            temporal_support_count: support_count.saturating_sub(explicit_support_count),
            unique_temporal_windows,
            effect_record_signature_variants: 1,
            positive_effect_signals: 0,
            negative_effect_signals: 0,
            counterevidence: 0,
            explicit_support_total_for_cause: explicit_support_count,
            explicit_effect_variants_for_cause: usize::from(explicit_support_count > 0),
            transition_lift: strength,
            temporal_consistency: 0.9,
            outcome_stability: 0.8,
            causal_strength: strength,
            state,
            last_updated: 0.0,
        }
    }

    // ── 1. Fresh engine is empty ──

    #[test]
    fn new_engine_is_empty() {
        let engine = PolicyEngine::new();
        assert!(engine.hints.is_empty());
        assert!(engine.key_index.is_empty());
    }

    // ── 2. No seeds → empty report ──

    #[test]
    fn no_seeds_produces_empty_report() {
        let mut engine = PolicyEngine::new();
        let causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let belief = BeliefEngine::default();
        let records = HashMap::new();

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.hints_found, 0);
        assert_eq!(report.seeds_found, 0);
    }

    // ── 3. Negative outcome produces Avoid/VerifyFirst ──

    #[test]
    fn negative_outcome_produces_avoid_or_verify() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Deployed untested code to production",
                "default",
                &["deploy"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Application crashed after deployment failure",
                "default",
                &["deploy", "incident"],
                "fact",
            ),
        );
        records.insert(
            "e2".into(),
            make_record(
                "e2",
                "Rollback was needed after the failure",
                "default",
                &["deploy", "incident"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            0.80,
            CausalState::Stable,
        );
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.hints_found, 1);

        let hint = engine.hints.values().next().unwrap();
        assert!(
            hint.action_kind == PolicyActionKind::Avoid
                || hint.action_kind == PolicyActionKind::VerifyFirst,
            "negative outcome should produce Avoid or VerifyFirst, got {:?}",
            hint.action_kind
        );
        assert!(
            hint.risk_score > 0.0,
            "risk_score should be positive for negative polarity"
        );
    }

    // ── 4. Positive outcome produces Prefer/Recommend ──

    #[test]
    fn positive_outcome_produces_prefer_or_recommend() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Enabled caching for API endpoints",
                "default",
                &["caching", "api"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "API response times improved significantly",
                "default",
                &["caching", "api"],
                "fact",
            ),
        );
        records.insert(
            "e2".into(),
            make_record(
                "e2",
                "Successfully deployed the optimized caching layer",
                "default",
                &["caching", "api"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            0.80,
            CausalState::Stable,
        );
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.hints_found, 1);

        let hint = engine.hints.values().next().unwrap();
        assert!(
            hint.action_kind == PolicyActionKind::Prefer
                || hint.action_kind == PolicyActionKind::Recommend,
            "positive outcome should produce Prefer or Recommend, got {:?}",
            hint.action_kind
        );
        assert!(
            hint.risk_score == 0.0,
            "risk_score should be 0 for positive polarity"
        );
    }

    // ── 5. Neutral polarity produces Warn ──

    #[test]
    fn neutral_polarity_produces_warn() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Changed the database schema version",
                "default",
                &["database"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Migration completed on schedule",
                "default",
                &["database"],
                "fact",
            ),
        );

        let pattern =
            make_causal_pattern("p1", "default", &["c1"], &["e1"], 0.80, CausalState::Stable);
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.hints_found, 1);

        let hint = engine.hints.values().next().unwrap();
        assert_eq!(
            hint.action_kind,
            PolicyActionKind::Warn,
            "neutral polarity should produce Warn"
        );
    }

    #[test]
    fn mixed_explicit_outcomes_do_not_produce_policy_hint() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Changed deploy workflow rollout",
                "default",
                &["deploy", "workflow"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Release stability improved after rollout change",
                "default",
                &["deploy", "stability", "improvement"],
                "fact",
            ),
        );
        records.insert(
            "e2".into(),
            make_record(
                "e2",
                "Rollback frequency increased after rollout change",
                "default",
                &["deploy", "rollback", "regression"],
                "fact",
            ),
        );
        records.insert(
            "e3".into(),
            make_record(
                "e3",
                "Security review load increased after rollout change",
                "default",
                &["deploy", "security", "review"],
                "fact",
            ),
        );

        let mut pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2", "e3"],
            &["b_cause"],
            &[],
            0.90,
            CausalState::Stable,
            9,
            9,
            1,
        );
        pattern.effect_record_signature_variants = 3;
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.seeds_found, 0);
        assert_eq!(report.hints_found, 0);
    }

    // ── 6. Weak causal patterns are not seeded ──

    #[test]
    fn weak_causal_not_seeded() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let belief = BeliefEngine::default();
        let records = HashMap::new();

        // Candidate with strength below MIN_CAUSAL_STRENGTH_FOR_SEED (0.65)
        let pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1"],
            0.40,
            CausalState::Candidate,
        );
        causal.patterns.insert("p1".into(), pattern);

        // Rejected pattern
        let pattern2 = make_causal_pattern(
            "p2",
            "default",
            &["c2"],
            &["e2"],
            0.30,
            CausalState::Rejected,
        );
        causal.patterns.insert("p2".into(), pattern2);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(
            report.seeds_found, 0,
            "weak/rejected patterns should not be seeded"
        );
        assert_eq!(report.hints_found, 0);
    }

    // ── 6b. Single-observation causal patterns are not seeded ──

    #[test]
    fn single_observation_causal_not_seeded() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            0.90,
            CausalState::Stable,
        );
        pattern.support_count = 1;
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &HashMap::new());
        assert_eq!(
            report.seeds_found, 0,
            "single-observation causal patterns should not seed policy hints"
        );
        assert_eq!(report.hints_found, 0);
    }

    #[test]
    fn strong_single_window_causal_not_seeded() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            &["b_cause"],
            &[],
            0.90,
            CausalState::Stable,
            3,
            0,
            1,
        );
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &HashMap::new());
        assert_eq!(report.seeds_found, 0);
        assert_eq!(report.hints_found, 0);
    }

    #[test]
    fn repeated_window_causal_can_seed_policy() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Enabled canary releases",
                "default",
                &["deploy"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Release stability improved after canary releases",
                "default",
                &["deploy", "improvement"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1"],
            &["b_cause"],
            &[],
            0.90,
            CausalState::Stable,
            3,
            0,
            2,
        );
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.seeds_found, 1);
        assert_eq!(report.hints_found, 1);
    }

    #[test]
    fn explicit_repeated_support_can_seed_policy() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Enabled automated rollback gate",
                "default",
                &["rollback"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Recovery improved after automated rollback gate",
                "default",
                &["rollback", "improvement"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1"],
            &["b_cause"],
            &[],
            0.88,
            CausalState::Stable,
            3,
            2,
            1,
        );
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.seeds_found, 1);
        assert_eq!(report.hints_found, 1);
    }

    #[test]
    fn high_counterevidence_causal_not_seeded() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Changed deploy workflow",
                "default",
                &["deploy"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Deploy stability improved after workflow change",
                "default",
                &["deploy", "improvement"],
                "fact",
            ),
        );

        let mut pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1"],
            &["b_cause"],
            &[],
            0.90,
            CausalState::Stable,
            3,
            2,
            2,
        );
        pattern.counterevidence = 4;
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.seeds_found, 0);
        assert_eq!(report.hints_found, 0);
    }

    #[test]
    fn bounded_counterevidence_causal_can_seed_policy() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record(
                "c1",
                "Enabled canary workflow",
                "default",
                &["deploy", "canary"],
                "decision",
            ),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Release stability improved after canary workflow",
                "default",
                &["deploy", "improvement"],
                "fact",
            ),
        );

        let mut pattern = make_causal_pattern_with_evidence(
            "p1",
            "default",
            &["c1"],
            &["e1"],
            &["b_cause"],
            &[],
            0.90,
            CausalState::Stable,
            3,
            2,
            2,
        );
        pattern.counterevidence = 2;
        causal.patterns.insert("p1".into(), pattern);

        let report = engine.discover(&causal, &concept, &belief, &records);
        assert_eq!(report.seeds_found, 1);
        assert_eq!(report.hints_found, 1);
    }

    // ── 7. Full rebuild clears state ──

    #[test]
    fn full_rebuild_clears_state() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record("c1", "Cause action", "default", &["ops"], "decision"),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Effect with failure and crash",
                "default",
                &["ops"],
                "fact",
            ),
        );
        records.insert(
            "e2".into(),
            make_record(
                "e2",
                "Another failure incident occurred",
                "default",
                &["ops"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            0.80,
            CausalState::Stable,
        );
        causal.patterns.insert("p1".into(), pattern);
        engine.discover(&causal, &concept, &belief, &records);
        assert!(!engine.hints.is_empty());

        // Second pass: empty causal
        let empty_causal = CausalEngine::new();
        let report = engine.discover(&empty_causal, &concept, &belief, &records);
        assert!(
            engine.hints.is_empty(),
            "full rebuild should clear old hints"
        );
        assert_eq!(report.hints_found, 0);
    }

    // ── 8. Provenance is complete ──

    #[test]
    fn provenance_is_complete() {
        let mut engine = PolicyEngine::new();
        let mut causal = CausalEngine::new();
        let concept = ConceptEngine::new();
        let mut belief = BeliefEngine::default();
        inject_singleton_belief(&mut belief, "b_cause");

        let mut records = HashMap::new();
        records.insert(
            "c1".into(),
            make_record("c1", "Root cause action", "default", &["ops"], "decision"),
        );
        records.insert(
            "e1".into(),
            make_record(
                "e1",
                "Effect with error and failure",
                "default",
                &["ops"],
                "fact",
            ),
        );
        records.insert(
            "e2".into(),
            make_record(
                "e2",
                "Another incident with crash",
                "default",
                &["ops"],
                "fact",
            ),
        );

        let pattern = make_causal_pattern(
            "p1",
            "default",
            &["c1"],
            &["e1", "e2"],
            0.80,
            CausalState::Stable,
        );
        causal.patterns.insert("p1".into(), pattern);

        engine.discover(&causal, &concept, &belief, &records);

        let hint = engine.hints.values().next().unwrap();
        assert!(
            !hint.trigger_causal_ids.is_empty(),
            "causal provenance required"
        );
        assert!(
            !hint.supporting_record_ids.is_empty(),
            "record provenance required"
        );
        assert_eq!(
            hint.supporting_record_ids.len(),
            3,
            "should have cause + effect records"
        );
    }

    // ── 9. PolicyState defaults ──

    #[test]
    fn policy_state_default_is_candidate() {
        assert_eq!(PolicyState::default(), PolicyState::Candidate);
    }

    // ── 10. Serialization roundtrip ──

    #[test]
    fn engine_serialization_roundtrip() {
        let engine = PolicyEngine::new();
        let json = serde_json::to_string(&engine).unwrap();
        let restored: PolicyEngine = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.hints.len(), engine.hints.len());
    }

    // ── 11. PolicyStore save/load ──

    #[test]
    fn store_save_and_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = PolicyStore::new(dir.path());

        let engine = PolicyEngine::new();
        store.save(&engine).expect("save");

        let loaded = store.load().expect("load");
        assert_eq!(loaded.hints.len(), 0);
    }

    // ── 12. High causal_strength + negative → Avoid ──

    #[test]
    fn high_negative_maps_to_avoid() {
        assert_eq!(
            PolicyEngine::map_action_kind(Polarity::Negative, 0.80),
            PolicyActionKind::Avoid
        );
    }

    // ── 13. Medium causal_strength + negative → VerifyFirst ──

    #[test]
    fn medium_negative_maps_to_verify() {
        assert_eq!(
            PolicyEngine::map_action_kind(Polarity::Negative, 0.65),
            PolicyActionKind::VerifyFirst
        );
    }

    // ── 14. High positive → Prefer ──

    #[test]
    fn high_positive_maps_to_prefer() {
        assert_eq!(
            PolicyEngine::map_action_kind(Polarity::Positive, 0.80),
            PolicyActionKind::Prefer
        );
    }

    // ── 15. Medium positive → Recommend ──

    #[test]
    fn medium_positive_maps_to_recommend() {
        assert_eq!(
            PolicyEngine::map_action_kind(Polarity::Positive, 0.65),
            PolicyActionKind::Recommend
        );
    }

    // ── Surface tests ──

    fn make_hint(
        id: &str,
        key: &str,
        ns: &str,
        domain: &str,
        action: PolicyActionKind,
        state: PolicyState,
        strength: f32,
        confidence: f32,
        risk: f32,
    ) -> PolicyHint {
        PolicyHint {
            id: id.to_string(),
            key: key.to_string(),
            namespace: ns.to_string(),
            domain: domain.to_string(),
            action_kind: action,
            recommendation: format!("Test recommendation for {}", domain),
            trigger_causal_ids: vec!["causal_1".to_string()],
            trigger_concept_ids: Vec::new(),
            trigger_belief_ids: vec!["belief_1".to_string()],
            supporting_record_ids: vec!["rec_1".to_string(), "rec_2".to_string()],
            cause_record_ids: vec!["rec_1".to_string()],
            confidence,
            utility_score: 0.5,
            risk_score: risk,
            policy_strength: strength,
            state,
            last_updated: 0.0,
        }
    }

    // ── 16. Stable hints are surfaced ──

    #[test]
    fn stable_hints_are_surfaced() {
        let mut engine = PolicyEngine::new();
        let hint = make_hint(
            "h1",
            "k1",
            "default",
            "deploy",
            PolicyActionKind::Prefer,
            PolicyState::Stable,
            0.80,
            0.75,
            0.0,
        );
        engine.hints.insert("h1".into(), hint);

        let surfaced = super::surface_policy_hints(&engine, None);
        assert_eq!(surfaced.len(), 1);
        assert_eq!(surfaced[0].state, "stable");
        assert_eq!(surfaced[0].action_kind, "prefer");
    }

    // ── 17. Strong candidates can be surfaced ──

    #[test]
    fn strong_candidates_can_be_surfaced() {
        let mut engine = PolicyEngine::new();
        let hint = make_hint(
            "h1",
            "k1",
            "default",
            "deploy",
            PolicyActionKind::Recommend,
            PolicyState::Candidate,
            0.75,
            0.60,
            0.0,
        );
        engine.hints.insert("h1".into(), hint);

        let surfaced = super::surface_policy_hints(&engine, None);
        assert_eq!(surfaced.len(), 1, "strong candidate should be surfaced");
        assert_eq!(surfaced[0].state, "candidate");
    }

    // ── 18. Suppressed hints are not surfaced ──

    #[test]
    fn suppressed_hints_are_not_surfaced() {
        let mut engine = PolicyEngine::new();
        let hint = make_hint(
            "h1",
            "k1",
            "default",
            "deploy",
            PolicyActionKind::Avoid,
            PolicyState::Suppressed,
            0.80,
            0.75,
            0.5,
        );
        engine.hints.insert("h1".into(), hint);

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(
            surfaced.is_empty(),
            "suppressed hints should not be surfaced"
        );
    }

    // ── 19. Rejected hints are not surfaced ──

    #[test]
    fn rejected_hints_are_not_surfaced() {
        let mut engine = PolicyEngine::new();
        let hint = make_hint(
            "h1",
            "k1",
            "default",
            "deploy",
            PolicyActionKind::Warn,
            PolicyState::Rejected,
            0.30,
            0.20,
            0.1,
        );
        engine.hints.insert("h1".into(), hint);

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(surfaced.is_empty(), "rejected hints should not be surfaced");
    }

    // ── 20. Hints without provenance are not surfaced ──

    #[test]
    fn hints_without_provenance_are_not_surfaced() {
        let mut engine = PolicyEngine::new();
        let mut hint = make_hint(
            "h1",
            "k1",
            "default",
            "deploy",
            PolicyActionKind::Prefer,
            PolicyState::Stable,
            0.80,
            0.75,
            0.0,
        );
        hint.trigger_causal_ids.clear(); // Remove provenance
        engine.hints.insert("h1".into(), hint);

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(
            surfaced.is_empty(),
            "hints without causal provenance should not be surfaced"
        );
    }

    // ── 21. Surface sorting is deterministic ──

    #[test]
    fn surface_sorting_is_deterministic() {
        let mut engine = PolicyEngine::new();
        engine.hints.insert(
            "h1".into(),
            make_hint(
                "h1",
                "k1",
                "default",
                "deploy",
                PolicyActionKind::Prefer,
                PolicyState::Stable,
                0.70,
                0.60,
                0.0,
            ),
        );
        engine.hints.insert(
            "h2".into(),
            make_hint(
                "h2",
                "k2",
                "default",
                "security",
                PolicyActionKind::Avoid,
                PolicyState::Stable,
                0.90,
                0.80,
                0.7,
            ),
        );
        engine.hints.insert(
            "h3".into(),
            make_hint(
                "h3",
                "k3",
                "default",
                "logging",
                PolicyActionKind::Recommend,
                PolicyState::Stable,
                0.80,
                0.70,
                0.0,
            ),
        );

        let s1 = super::surface_policy_hints(&engine, None);
        let s2 = super::surface_policy_hints(&engine, None);

        assert_eq!(s1.len(), 3);
        // Same order both times
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.id, b.id, "sorting should be deterministic");
        }
        // Highest strength first
        assert_eq!(s1[0].id, "h2", "highest strength should be first");
    }

    #[test]
    fn policy_hint_id_is_deterministic_from_key() {
        let id1 = super::deterministic_id("default:prefer:default:cause→effect");
        let id2 = super::deterministic_id("default:prefer:default:cause→effect");
        let id3 = super::deterministic_id("default:avoid:default:cause→effect");

        assert_eq!(id1, id2, "same policy key must yield same id");
        assert_ne!(id1, id3, "different policy key must yield different id");
    }

    // ── 22. Surface limit is enforced ──

    #[test]
    fn surface_limit_is_enforced() {
        let mut engine = PolicyEngine::new();
        for i in 0..15 {
            let domain = format!("domain_{}", i);
            engine.hints.insert(
                format!("h{}", i),
                make_hint(
                    &format!("h{}", i),
                    &format!("k{}", i),
                    "default",
                    &domain,
                    PolicyActionKind::Prefer,
                    PolicyState::Stable,
                    0.80,
                    0.70,
                    0.0,
                ),
            );
        }

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(
            surfaced.len() <= 10,
            "should respect MAX_SURFACED_HINTS, got {}",
            surfaced.len()
        );

        let surfaced_5 = super::surface_policy_hints(&engine, Some(5));
        assert!(
            surfaced_5.len() <= 5,
            "should respect explicit limit, got {}",
            surfaced_5.len()
        );
    }

    // ── 23. Per-domain cap enforced ──

    #[test]
    fn per_domain_cap_enforced() {
        let mut engine = PolicyEngine::new();
        for i in 0..6 {
            engine.hints.insert(
                format!("h{}", i),
                make_hint(
                    &format!("h{}", i),
                    &format!("k{}", i),
                    "default",
                    "deploy",
                    PolicyActionKind::Prefer,
                    PolicyState::Stable,
                    0.80 - i as f32 * 0.01,
                    0.70,
                    0.0,
                ),
            );
        }

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(
            surfaced.len() <= 3,
            "should respect MAX_SURFACED_PER_DOMAIN=3, got {}",
            surfaced.len()
        );
    }

    // ── 24. Weak candidates not surfaced ──

    #[test]
    fn weak_candidates_not_surfaced() {
        let mut engine = PolicyEngine::new();
        // Candidate with strength below STRONG_CANDIDATE_THRESHOLD (0.70)
        engine.hints.insert(
            "h1".into(),
            make_hint(
                "h1",
                "k1",
                "default",
                "deploy",
                PolicyActionKind::Recommend,
                PolicyState::Candidate,
                0.50,
                0.60,
                0.0,
            ),
        );

        let surfaced = super::surface_policy_hints(&engine, None);
        assert!(
            surfaced.is_empty(),
            "weak candidates should not be surfaced"
        );
    }

    // ── 25. Surfaced hints have full provenance ──

    #[test]
    fn surfaced_hints_have_full_provenance() {
        let mut engine = PolicyEngine::new();
        engine.hints.insert(
            "h1".into(),
            make_hint(
                "h1",
                "k1",
                "default",
                "deploy",
                PolicyActionKind::Prefer,
                PolicyState::Stable,
                0.80,
                0.75,
                0.0,
            ),
        );

        let surfaced = super::surface_policy_hints(&engine, None);
        assert_eq!(surfaced.len(), 1);
        let h = &surfaced[0];
        assert!(
            !h.trigger_causal_ids.is_empty(),
            "must have causal provenance"
        );
        assert!(
            !h.supporting_record_ids.is_empty(),
            "must have record provenance"
        );
    }
}
