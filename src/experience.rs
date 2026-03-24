//! Autonomous Cognitive Plasticity — Experience Ingestion Layer (v5)
//!
//! Implements the closed inference loop:
//!   capture_experience() → extract_experience_events() → apply_experience()
//!
//! The model stays frozen. This layer observes model output and lets the
//! cognitive substrate evolve in response — without changing model weights.
//!
//! Safety contract (Phase 3.1 guards, always enforced):
//!   - ModelInference records: max confidence = 0.70 (GENERATED_CONFIDENCE_CEILING)
//!   - ModelInference records: max level = Domain (never Identity)
//!   - ModelInference contradicting recorded → recorded wins, generated gets volatility
//!   - PlasticityMode::Off (default) → capture_experience() is a no-op

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

/// Hard ceiling on confidence for model-generated records.
/// Cannot be raised without human confirmation (recorded/retrieved source).
pub const GENERATED_CONFIDENCE_CEILING: f32 = 0.70;

/// Base confidence for Asserted claims from inference.
pub const ASSERTED_BASE_CONFIDENCE: f32 = 0.50;

/// Base confidence for Hedged claims from inference.
pub const HEDGED_BASE_CONFIDENCE: f32 = 0.40;

/// Base confidence for Speculative claims from inference.
pub const SPECULATIVE_BASE_CONFIDENCE: f32 = 0.30;

/// SDR Tanimoto threshold above which a sentence is treated as a Confirmation
/// of an existing belief rather than a new Claim.
pub const CONFIRMATION_TANIMOTO_THRESHOLD: f32 = 0.65;

/// SDR Tanimoto threshold above which a sentence is compared for contradiction.
pub const CONTRADICTION_TANIMOTO_THRESHOLD: f32 = 0.50;

/// Max new records stored per single capture_experience() call (Default policy).
pub const DEFAULT_MAX_NEW_RECORDS: u8 = 3;

/// How many contradiction events in one session trigger a HallucinationAlert.
pub const HALLUCINATION_ALERT_THRESHOLD: u32 = 3;

/// Volatility bump applied to a generated record when it contradicts an
/// existing recorded/retrieved belief.
pub const CONTRADICTION_VOLATILITY_DELTA: f32 = 0.15;

/// Confidence nudge applied when a generated claim confirms an existing belief.
pub const CONFIRMATION_STRENGTH_DELTA: f32 = 0.02;

// ── Experience source ────────────────────────────────────────────────────────

/// Describes the *channel* through which an experience arrived.
///
/// Stored in `Record.metadata["experience_source"]` — does **not** replace
/// `source_type`, which describes the nature of the record itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperienceSource {
    /// External document, knowledge base, tool result.
    WorldFact,
    /// The human stated this directly.
    HumanStatement,
    /// The model produced this during inference.
    ModelInference,
    /// Aura formed this internally during maintenance.
    InternalHypothesis,
}

impl ExperienceSource {
    /// Maps to the `source_type` string used by `Record`.
    pub fn to_source_type(&self) -> &'static str {
        match self {
            Self::WorldFact => "recorded",
            Self::HumanStatement => "recorded",
            Self::ModelInference => "generated",
            Self::InternalHypothesis => "inferred",
        }
    }

    /// Base confidence for records from this source.
    pub fn base_confidence(&self) -> f32 {
        match self {
            Self::WorldFact => 0.90,
            Self::HumanStatement => 0.90,
            Self::ModelInference => ASSERTED_BASE_CONFIDENCE,
            Self::InternalHypothesis => 0.60,
        }
    }

    /// Metadata key stored in `Record.metadata`.
    pub fn metadata_value(&self) -> &'static str {
        match self {
            Self::WorldFact => "world_fact",
            Self::HumanStatement => "human",
            Self::ModelInference => "model_inference",
            Self::InternalHypothesis => "internal",
        }
    }
}

// ── Claim certainty ──────────────────────────────────────────────────────────

/// How certain the model appears when making a claim.
/// Detected heuristically from hedge markers in the sentence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimCertainty {
    /// "X може бути Y", "possibly", "might" — least certain.
    Speculative,
    /// "X, мабуть, Y", "probably", "likely", "здається" — hedged.
    Hedged,
    /// Direct assertion without hedging markers — most certain.
    Asserted,
}

impl ClaimCertainty {
    pub fn base_confidence(&self) -> f32 {
        match self {
            Self::Asserted => ASSERTED_BASE_CONFIDENCE,
            Self::Hedged => HEDGED_BASE_CONFIDENCE,
            Self::Speculative => SPECULATIVE_BASE_CONFIDENCE,
        }
    }
}

// ── Conflict severity ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictSeverity {
    /// Tanimoto overlap only — may be coincidental.
    Weak,
    /// Semantic contradiction keyword detected in context.
    Strong,
}

// ── Experience events ────────────────────────────────────────────────────────

/// A single structured event extracted from a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExperienceEvent {
    /// A new claim made by the model — not yet in memory.
    Claim {
        text: String,
        tags: Vec<String>,
        /// "fact" | "decision" | "preference" | "trend"
        semantic_type: String,
        certainty: ClaimCertainty,
    },
    /// The model's statement is close enough to an existing belief to
    /// reinforce it.
    Confirmation {
        /// Belief that is being reinforced.
        belief_id: String,
        /// How much to nudge confidence upward.
        strength_delta: f32,
    },
    /// The model's statement contradicts an existing belief.
    Contradiction {
        /// Belief that conflicts with this statement.
        belief_id: String,
        /// Volatility increase to apply to affected records.
        volatility_delta: f32,
        severity: ConflictSeverity,
    },
    /// The model hedged or expressed uncertainty about something it said.
    UncertaintyMarker {
        text: String,
        /// If this clearly refers to a specific record.
        affects_record_id: Option<String>,
    },
    /// The model made a commitment or stated a plan.
    Commitment {
        text: String,
        tags: Vec<String>,
        deadline_hint: Option<String>,
    },
}

// ── Plasticity policy ────────────────────────────────────────────────────────

/// Controls what apply_experience() is allowed to do.
///
/// Conservative defaults: new records allowed but capped at 3,
/// connection weakening and correction candidates disabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityPolicy {
    /// Whether new Claim events may be stored as records.
    pub allow_new_records: bool,
    /// Hard cap on new records per call.
    pub max_new_records_per_call: u8,
    /// Whether Confirmation events may nudge belief confidence.
    pub allow_belief_reinforcement: bool,
    /// Whether Contradiction events may raise record volatility.
    pub allow_volatility_increase: bool,
    /// Whether apply_experience() may weaken existing connections.
    pub allow_connection_weakening: bool,
    /// Whether strong Contradictions may open a correction candidate.
    pub allow_correction_candidates: bool,
    /// Minimum certainty level for a Claim to be stored.
    /// Speculative claims are skipped unless this is set to Speculative.
    pub min_claim_certainty: ClaimCertainty,
}

impl Default for PlasticityPolicy {
    fn default() -> Self {
        Self {
            allow_new_records: true,
            max_new_records_per_call: DEFAULT_MAX_NEW_RECORDS,
            allow_belief_reinforcement: true,
            allow_volatility_increase: true,
            allow_connection_weakening: false,
            allow_correction_candidates: false,
            min_claim_certainty: ClaimCertainty::Hedged,
        }
    }
}

impl PlasticityPolicy {
    /// A fully permissive policy. Use only in tests or explicit operator opt-in.
    pub fn full() -> Self {
        Self {
            allow_new_records: true,
            max_new_records_per_call: 10,
            allow_belief_reinforcement: true,
            allow_volatility_increase: true,
            allow_connection_weakening: true,
            allow_correction_candidates: true,
            min_claim_certainty: ClaimCertainty::Speculative,
        }
    }

    /// No mutations — only logging. Used with PlasticityMode::Observe.
    pub fn observe_only() -> Self {
        Self {
            allow_new_records: false,
            max_new_records_per_call: 0,
            allow_belief_reinforcement: false,
            allow_volatility_increase: false,
            allow_connection_weakening: false,
            allow_correction_candidates: false,
            min_claim_certainty: ClaimCertainty::Asserted,
        }
    }
}

// ── Plasticity mode ──────────────────────────────────────────────────────────

/// Controls whether and how experience is applied.
///
/// Default: Off — the system never changes silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PlasticityMode {
    /// capture_experience() is a no-op. System is read-only from inference.
    Off = 0,
    /// Events are extracted and logged but never applied.
    Observe = 1,
    /// Applied with Default PlasticityPolicy.
    Limited = 2,
    /// Applied with operator-supplied PlasticityPolicy.
    Full = 3,
}

impl PlasticityMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Observe,
            2 => Self::Limited,
            3 => Self::Full,
            _ => Self::Off,
        }
    }
}

// ── Plasticity report ────────────────────────────────────────────────────────

/// Outcome of a single apply_experience() call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlasticityReport {
    pub events_processed: usize,
    pub new_records_stored: usize,
    pub beliefs_reinforced: usize,
    pub volatility_increases: usize,
    pub connections_weakened: usize,
    pub correction_candidates_opened: usize,
    /// Events that did not pass policy filters.
    pub events_skipped: usize,
    pub skipped_reasons: Vec<String>,
    /// Incremented each time a generated claim contradicts a stable recorded belief.
    pub hallucination_alerts: u32,
}

// ── Plasticity audit entry ───────────────────────────────────────────────────

/// One entry in the persistent audit trail of experience-driven mutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityAuditEntry {
    pub timestamp: u64,
    pub session_id: String,
    /// "new_record" | "belief_reinforced" | "volatility_raised" | "skipped"
    pub event_kind: String,
    /// Record ID, belief ID, or empty string.
    pub target_id: String,
    /// xxh3 hash of the prompt that triggered this event.
    pub source_prompt_hash: String,
    pub confidence_before: Option<f32>,
    pub confidence_after: Option<f32>,
    /// Serialized PlasticityPolicy name ("default" | "observe" | "full" | "custom").
    pub policy_name: String,
}

// ── Experience capture ───────────────────────────────────────────────────────

/// The result of one capture_experience() call.
///
/// Contains extracted events and the plasticity report from applying them.
/// All mutations that happened are reflected in plasticity_report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceCapture {
    pub session_id: String,
    pub timestamp: u64,
    /// xxh3 hash of the prompt — used to deduplicate identical calls.
    pub prompt_hash: String,
    /// First 200 chars of model_response for audit readability.
    pub response_summary: String,
    /// Record IDs that were in the preamble context for this turn.
    pub context_record_ids: Vec<String>,
    pub source: ExperienceSource,
    /// Extracted events (before policy filtering).
    pub raw_events: Vec<ExperienceEvent>,
    /// What was actually applied.
    pub plasticity_report: PlasticityReport,
}

// ── Heuristic extractor ──────────────────────────────────────────────────────

/// English + Ukrainian hedge markers indicating Hedged certainty.
static HEDGE_MARKERS: &[&str] = &[
    // Ukrainian
    "мабуть",
    "здається",
    "можливо",
    "схоже",
    "напевно",
    "мабуть",
    "не впевнений",
    "не знаю",
    "не певен",
    "очевидно",
    // English
    "probably",
    "likely",
    "seems",
    "appears",
    "might",
    "may",
    "perhaps",
    "possibly",
    "i think",
    "i believe",
    "not sure",
    "uncertain",
    "arguably",
];

/// Markers indicating Speculative certainty.
static SPECULATIVE_MARKERS: &[&str] = &[
    // Ukrainian
    "може бути",
    "можливо що",
    "теоретично",
    "припускаю",
    // English
    "might be",
    "could be",
    "possibly",
    "theoretically",
    "i suppose",
    "i guess",
    "speculate",
];

/// Contradiction indicator keywords (used together with SDR overlap).
static CONTRADICTION_KEYWORDS: &[&str] = &[
    // Ukrainian
    "але",
    "однак",
    "проте",
    "насправді",
    "навпаки",
    "не є",
    "не буде",
    "хибно",
    "помилково",
    // English
    "but",
    "however",
    "actually",
    "contrary",
    "incorrect",
    "wrong",
    "not true",
    "false",
    "instead",
    "rather",
    "on the contrary",
];

/// Promise / commitment markers.
static COMMITMENT_MARKERS: &[&str] = &[
    // Ukrainian
    "зроблю",
    "буду",
    "планую",
    "збираюсь",
    "обіцяю",
    "зобов'яжусь",
    "встановлю",
    "запущу",
    "виправлю",
    // English
    "will",
    "i'll",
    "going to",
    "plan to",
    "intend to",
    "promise",
    "shall",
    "i will",
    "let me",
];

/// Detect certainty level from a sentence.
fn detect_certainty(sentence: &str) -> ClaimCertainty {
    let lower = sentence.to_lowercase();
    if SPECULATIVE_MARKERS.iter().any(|m| lower.contains(m)) {
        return ClaimCertainty::Speculative;
    }
    if HEDGE_MARKERS.iter().any(|m| lower.contains(m)) {
        return ClaimCertainty::Hedged;
    }
    ClaimCertainty::Asserted
}

/// Detect if a sentence is a commitment/plan.
fn is_commitment(sentence: &str) -> bool {
    let lower = sentence.to_lowercase();
    COMMITMENT_MARKERS.iter().any(|m| lower.contains(m))
}

/// Extract deadline hints from a commitment sentence.
fn extract_deadline_hint(sentence: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    // Simple heuristic: look for time expressions
    let time_markers = [
        "today",
        "tomorrow",
        "next week",
        "this week",
        "by friday",
        "сьогодні",
        "завтра",
        "наступного тижня",
        "цього тижня",
    ];
    for marker in &time_markers {
        if lower.contains(marker) {
            return Some(marker.to_string());
        }
    }
    None
}

/// Infer basic tags from sentence content (very conservative).
fn infer_tags(sentence: &str) -> Vec<String> {
    let lower = sentence.to_lowercase();
    let mut tags = Vec::new();
    // Domain hints
    let domain_hints: &[(&str, &str)] = &[
        ("health", "health"),
        ("medical", "medical"),
        ("code", "code"),
        ("error", "debug"),
        ("bug", "debug"),
        ("memory", "memory"),
        ("deploy", "ops"),
        ("server", "ops"),
        ("database", "data"),
        ("здоров", "health"),
        ("код", "code"),
        ("помилка", "debug"),
    ];
    for (keyword, tag) in domain_hints {
        if lower.contains(keyword) {
            let tag_str = tag.to_string();
            if !tags.contains(&tag_str) {
                tags.push(tag_str);
            }
        }
    }
    tags
}

/// Infer semantic_type from a sentence.
fn infer_semantic_type(sentence: &str, certainty: &ClaimCertainty) -> &'static str {
    let lower = sentence.to_lowercase();
    if is_commitment(sentence) {
        return "decision";
    }
    if lower.contains("prefer")
        || lower.contains("like")
        || lower.contains("перевагу")
        || lower.contains("подобається")
    {
        return "preference";
    }
    if lower.contains("trend")
        || lower.contains("pattern")
        || lower.contains("завжди")
        || lower.contains("зазвичай")
        || lower.contains("often")
    {
        return "trend";
    }
    if *certainty == ClaimCertainty::Speculative {
        return "fact"; // store speculatives as facts with low confidence
    }
    "fact"
}

/// Split text into sentences (simple heuristic).
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| s.split_whitespace().count() >= 4) // skip fragments < 4 words
        .collect()
}

/// Core heuristic extractor — Variant A (no SDR, no LLM dependency).
///
/// Processes model_response into ExperienceEvents using only text patterns.
/// Variant B (SDR similarity against existing records) is applied on top
/// inside Aura::capture_experience() where record access is available.
pub fn extract_experience_events_heuristic(response: &str) -> Vec<ExperienceEvent> {
    let sentences = split_sentences(response);
    let mut events = Vec::new();

    for sentence in &sentences {
        let lower = sentence.to_lowercase();

        // Commitment detection (higher priority — check before claim)
        if is_commitment(sentence) {
            let deadline = extract_deadline_hint(sentence);
            let tags = infer_tags(sentence);
            events.push(ExperienceEvent::Commitment {
                text: sentence.clone(),
                tags,
                deadline_hint: deadline,
            });
            continue;
        }

        // Uncertainty marker detection
        let is_uncertain = HEDGE_MARKERS.iter().any(|m| lower.contains(m))
            || SPECULATIVE_MARKERS.iter().any(|m| lower.contains(m));
        let has_contradiction_keyword = CONTRADICTION_KEYWORDS.iter().any(|m| lower.contains(m));

        if is_uncertain && has_contradiction_keyword {
            events.push(ExperienceEvent::UncertaintyMarker {
                text: sentence.clone(),
                affects_record_id: None,
            });
            continue;
        }

        // Claim detection
        let certainty = detect_certainty(sentence);
        let semantic_type = infer_semantic_type(sentence, &certainty).to_string();
        let tags = infer_tags(sentence);

        events.push(ExperienceEvent::Claim {
            text: sentence.clone(),
            tags,
            semantic_type,
            certainty,
        });
    }

    events
}

// ── Anti-hallucination guards (Phase 3.1) ───────────────────────────────────

/// Guard 1: Clamp confidence of a generated record to the ceiling.
///
/// Called before storing any ModelInference record.
pub fn apply_confidence_ceiling(confidence: f32) -> f32 {
    confidence.min(GENERATED_CONFIDENCE_CEILING)
}

/// Guard 2 + 3: Check whether a ModelInference claim should be blocked
/// because it contradicts a stable recorded/retrieved belief.
///
/// Returns (allow_store, volatility_delta_for_generated).
/// When blocked: the generated record is NOT stored; existing belief gets
/// volatility bump instead.
pub fn apply_contradiction_asymmetry(
    claim_certainty: &ClaimCertainty,
    contradicts_recorded_belief: bool,
) -> (bool, f32) {
    if contradicts_recorded_belief {
        // Recorded wins — do not store generated, raise its volatility instead.
        (false, CONTRADICTION_VOLATILITY_DELTA)
    } else {
        let _ = claim_certainty; // no restriction for generated-vs-generated
        (true, 0.0)
    }
}

/// Guard 4: ModelInference records must not exceed Level::Domain.
///
/// Returns the clamped level value. Identity (4) is reduced to Domain (3).
pub fn clamp_generated_level(level_value: u8) -> u8 {
    // Level::Identity == 4 in the existing enum ordering.
    // Level::Domain   == 3.
    level_value.min(3)
}

// ── Plasticity risk assessment (Phase 3.2) ───────────────────────────────────

/// Risk level derived from accumulated plasticity telemetry.
///
/// Controls automatic throttling of new record creation and can trigger
/// operator alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlasticityRisk {
    /// System is evolving safely — no intervention needed.
    Safe,
    /// Elevated contradictions or generated ratio — operator should monitor.
    Monitor,
    /// High contradiction rate — max_new_records_per_call auto-reduced to 1.
    Restrict,
    /// Critical hallucination rate — allow_new_records auto-set to false.
    Pause,
}

/// Thresholds used by `PlasticityRiskAssessment::compute()`.
pub const RISK_GENERATED_RATIO_MONITOR: f32 = 0.70;
pub const RISK_GENERATED_RATIO_RESTRICT: f32 = 0.85;
pub const RISK_CONTRADICTION_RATE_MONITOR: f32 = 0.15;
pub const RISK_CONTRADICTION_RATE_RESTRICT: f32 = 0.30;
pub const RISK_HALLUCINATION_ALERTS_RESTRICT: u32 = 5;
pub const RISK_HALLUCINATION_ALERTS_PAUSE: u32 = 10;

/// Snapshot of current plasticity health — computed on demand from
/// accumulated `PlasticityReport` data across recent cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityRiskAssessment {
    /// Composite risk score in [0.0, 1.0].
    pub score: f32,
    /// Fraction of total records that come from ModelInference.
    pub generated_ratio: f32,
    /// Fraction of events that were contradictions this window.
    pub contradiction_rate: f32,
    /// Total hallucination alerts accumulated since last reset.
    pub hallucination_alerts: u32,
    pub risk: PlasticityRisk,
    /// Human-readable recommendation for the operator.
    pub recommendation: String,
}

impl PlasticityRiskAssessment {
    /// Compute a risk assessment from accumulated telemetry.
    ///
    /// - `total_records`: current record count in the store.
    /// - `generated_records`: records with source_type == "generated".
    /// - `events_processed_window`: total events seen in recent window.
    /// - `contradictions_window`: contradiction events in the same window.
    /// - `hallucination_alerts_total`: cumulative hallucination alert count.
    pub fn compute(
        total_records: usize,
        generated_records: usize,
        events_processed_window: usize,
        contradictions_window: usize,
        hallucination_alerts_total: u32,
    ) -> Self {
        let generated_ratio = if total_records == 0 {
            0.0
        } else {
            generated_records as f32 / total_records as f32
        };

        let contradiction_rate = if events_processed_window == 0 {
            0.0
        } else {
            contradictions_window as f32 / events_processed_window as f32
        };

        // Composite score: weighted average of three signals.
        // generated_ratio: 0.40, contradiction_rate: 0.35, hallucination: 0.25
        let hallucination_score =
            (hallucination_alerts_total as f32 / RISK_HALLUCINATION_ALERTS_PAUSE as f32).min(1.0);
        let score = 0.40 * generated_ratio.min(1.0)
            + 0.35 * contradiction_rate.min(1.0)
            + 0.25 * hallucination_score;

        let risk = if hallucination_alerts_total >= RISK_HALLUCINATION_ALERTS_PAUSE {
            PlasticityRisk::Pause
        } else if hallucination_alerts_total >= RISK_HALLUCINATION_ALERTS_RESTRICT
            || contradiction_rate >= RISK_CONTRADICTION_RATE_RESTRICT
            || generated_ratio >= RISK_GENERATED_RATIO_RESTRICT
        {
            PlasticityRisk::Restrict
        } else if contradiction_rate >= RISK_CONTRADICTION_RATE_MONITOR
            || generated_ratio >= RISK_GENERATED_RATIO_MONITOR
        {
            PlasticityRisk::Monitor
        } else {
            PlasticityRisk::Safe
        };

        let recommendation = match risk {
            PlasticityRisk::Safe => "Plasticity is operating within normal bounds.".to_string(),
            PlasticityRisk::Monitor => format!(
                "Elevated signal (generated_ratio={:.0}%, contradiction_rate={:.0}%). \
                 Review captured experiences before next maintenance cycle.",
                generated_ratio * 100.0,
                contradiction_rate * 100.0
            ),
            PlasticityRisk::Restrict => format!(
                "High contradiction rate or generated saturation \
                 (alerts={}, contradiction_rate={:.0}%). \
                 max_new_records_per_call auto-reduced to 1.",
                hallucination_alerts_total,
                contradiction_rate * 100.0
            ),
            PlasticityRisk::Pause => format!(
                "Critical hallucination volume (alerts={}). \
                 New record creation suspended. \
                 Operator must clear alerts and reset to resume.",
                hallucination_alerts_total
            ),
        };

        Self {
            score,
            generated_ratio,
            contradiction_rate,
            hallucination_alerts: hallucination_alerts_total,
            risk,
            recommendation,
        }
    }

    /// Apply the risk-driven throttling to a `PlasticityPolicy` in place.
    ///
    /// Called at the start of each `capture_experience()` when mode is
    /// Limited or Full.  Returns the (possibly modified) policy.
    pub fn apply_throttling(&self, mut policy: PlasticityPolicy) -> PlasticityPolicy {
        match self.risk {
            PlasticityRisk::Safe | PlasticityRisk::Monitor => {
                // No change — let operator-supplied policy stand.
            }
            PlasticityRisk::Restrict => {
                // Reduce new records per call to 1.
                policy.max_new_records_per_call = policy.max_new_records_per_call.min(1);
            }
            PlasticityRisk::Pause => {
                // Suspend all new record creation.
                policy.allow_new_records = false;
                policy.max_new_records_per_call = 0;
            }
        }
        policy
    }
}

// ── Purge report (Phase 4.2) ─────────────────────────────────────────────────

/// Result of a `purge_inference_records()` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeReport {
    /// Total records examined.
    pub examined: usize,
    /// Records removed.
    pub removed: usize,
    /// Namespace filter that was applied (None = all namespaces).
    pub namespace_filter: Option<String>,
    /// Timestamp cutoff used (None = no cutoff).
    pub before_timestamp: Option<u64>,
}

// ── Experience queue ─────────────────────────────────────────────────────────

/// Thread-safe queue of pending ExperienceCaptures awaiting the next
/// maintenance cycle. Drained by MaintenanceService during phase 3.6.
#[derive(Default)]
pub struct ExperienceQueue {
    inner: std::sync::Mutex<Vec<ExperienceCapture>>,
}

impl ExperienceQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a batch of captures.
    pub fn enqueue(&self, captures: Vec<ExperienceCapture>) {
        let mut q = self.inner.lock().expect("experience queue poisoned");
        q.extend(captures);
    }

    /// Drain all pending captures (called at the start of maintenance phase 3.6).
    pub fn drain(&self) -> Vec<ExperienceCapture> {
        let mut q = self.inner.lock().expect("experience queue poisoned");
        std::mem::take(&mut *q)
    }

    /// Number of pending captures.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("experience queue poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Prompt hashing ───────────────────────────────────────────────────────────

/// Fast non-cryptographic hash of prompt text (xxh3 via std).
/// Used to deduplicate identical prompt→response pairs.
pub fn hash_prompt(prompt: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prompt.hash(&mut h);
    format!("{:016x}", h.finish())
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Current Unix timestamp in seconds.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certainty_detection_asserted() {
        let certainty = detect_certainty("The server is running on port 8080.");
        assert_eq!(certainty, ClaimCertainty::Asserted);
    }

    #[test]
    fn test_certainty_detection_hedged() {
        let certainty = detect_certainty("This probably causes a memory leak.");
        assert_eq!(certainty, ClaimCertainty::Hedged);
    }

    #[test]
    fn test_certainty_detection_speculative() {
        let certainty = detect_certainty("This could be a race condition.");
        assert_eq!(certainty, ClaimCertainty::Speculative);
    }

    #[test]
    fn test_commitment_detection() {
        assert!(is_commitment("I will fix the bug tomorrow."));
        assert!(is_commitment("I'll deploy the update next week."));
        assert!(!is_commitment("The server is running fine."));
    }

    #[test]
    fn test_heuristic_extractor_produces_events() {
        let response = "The deployment failed due to a missing config file. \
            I will fix the configuration tomorrow. \
            This might be related to the recent update.";
        let events = extract_experience_events_heuristic(response);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_heuristic_extractor_commitment_event() {
        let response = "I will restart the service after the fix.";
        let events = extract_experience_events_heuristic(response);
        assert!(events
            .iter()
            .any(|e| matches!(e, ExperienceEvent::Commitment { .. })));
    }

    #[test]
    fn test_confidence_ceiling_enforced() {
        assert_eq!(apply_confidence_ceiling(0.95), GENERATED_CONFIDENCE_CEILING);
        assert_eq!(apply_confidence_ceiling(0.60), 0.60);
        assert_eq!(apply_confidence_ceiling(0.70), 0.70);
    }

    #[test]
    fn test_contradiction_asymmetry_recorded_wins() {
        let (allow, volatility) = apply_contradiction_asymmetry(&ClaimCertainty::Asserted, true);
        assert!(
            !allow,
            "generated should be blocked when contradicting recorded"
        );
        assert!(volatility > 0.0, "volatility bump should be applied");
    }

    #[test]
    fn test_contradiction_asymmetry_generated_vs_generated() {
        let (allow, volatility) = apply_contradiction_asymmetry(&ClaimCertainty::Hedged, false);
        assert!(allow, "generated-vs-generated should be allowed");
        assert_eq!(volatility, 0.0);
    }

    #[test]
    fn test_clamp_generated_level_blocks_identity() {
        assert_eq!(clamp_generated_level(4), 3); // Identity → Domain
        assert_eq!(clamp_generated_level(3), 3); // Domain stays Domain
        assert_eq!(clamp_generated_level(2), 2); // Working stays Working
    }

    #[test]
    fn test_experience_queue_drain() {
        let q = ExperienceQueue::new();
        assert!(q.is_empty());
        q.enqueue(vec![ExperienceCapture {
            session_id: "s1".into(),
            timestamp: 0,
            prompt_hash: "abc".into(),
            response_summary: "test".into(),
            context_record_ids: vec![],
            source: ExperienceSource::ModelInference,
            raw_events: vec![],
            plasticity_report: PlasticityReport::default(),
        }]);
        assert_eq!(q.len(), 1);
        let drained = q.drain();
        assert_eq!(drained.len(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_experience_source_metadata_values() {
        assert_eq!(
            ExperienceSource::ModelInference.metadata_value(),
            "model_inference"
        );
        assert_eq!(ExperienceSource::HumanStatement.metadata_value(), "human");
        assert_eq!(ExperienceSource::WorldFact.to_source_type(), "recorded");
        assert_eq!(
            ExperienceSource::ModelInference.to_source_type(),
            "generated"
        );
    }

    #[test]
    fn test_plasticity_policy_default_is_conservative() {
        let p = PlasticityPolicy::default();
        assert!(p.allow_new_records);
        assert_eq!(p.max_new_records_per_call, DEFAULT_MAX_NEW_RECORDS);
        assert!(!p.allow_connection_weakening);
        assert!(!p.allow_correction_candidates);
        assert_eq!(p.min_claim_certainty, ClaimCertainty::Hedged);
    }

    #[test]
    fn test_plasticity_policy_observe_only_no_mutations() {
        let p = PlasticityPolicy::observe_only();
        assert!(!p.allow_new_records);
        assert!(!p.allow_belief_reinforcement);
        assert!(!p.allow_volatility_increase);
    }

    #[test]
    fn test_sentence_splitter_filters_fragments() {
        let text = "OK. This is a real sentence about memory. Short.";
        let sentences = split_sentences(text);
        // "OK" and "Short" should be filtered (< 4 words)
        assert!(sentences.iter().all(|s| s.split_whitespace().count() >= 4));
    }

    #[test]
    fn test_hash_prompt_deterministic() {
        let h1 = hash_prompt("same prompt");
        let h2 = hash_prompt("same prompt");
        let h3 = hash_prompt("different prompt");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
