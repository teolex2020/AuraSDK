//! Central gate registry for core stability and safety thresholds.
//!
//! Start small: this module captures the highest-signal P0 thresholds that are
//! currently reused across replay and cross-layer evaluation. Wider migration of
//! the test suite can build on top of this.

/// Immutable registry of core acceptance thresholds.
pub struct GateRegistry;

impl GateRegistry {
    pub const BELIEF_CHURN_SHADOW_MAX: f32 = 0.15;
    pub const BELIEF_CHURN_STABLE_MAX: f32 = 0.10;

    pub const CONCEPT_FALSE_MERGE_MAX: f32 = 0.05;
    pub const CONCEPT_MISLEADING_MAX: f32 = 0.05;
    pub const CONCEPT_CHURN_SHADOW_MAX: f32 = 0.20;
    pub const CONCEPT_CHURN_RELAXED_MAX: f32 = 0.25;
    pub const CONCEPT_CHURN_STABLE_MAX: f32 = 0.15;

    pub const CAUSAL_CHURN_SHADOW_MAX: f32 = 0.20;
    pub const CAUSAL_CHURN_RELAXED_MAX: f32 = 0.25;
    pub const CAUSAL_CHURN_STABLE_MAX: f32 = 0.15;

    pub const POLICY_CHURN_SHADOW_MAX: f32 = 0.20;
    pub const POLICY_CHURN_RELAXED_MAX: f32 = 0.25;
    pub const POLICY_CHURN_STABLE_MAX: f32 = 0.15;

    pub const CONCEPT_REPLAY_COUNT_DRIFT_MAX: usize = 1;
    pub const CONCEPT_REPLAY_CHURN_BASELINE_MAX: usize = 4;
    pub const CONCEPT_REPLAY_CHURN_ADAPTIVE_MAX: usize = 7;
    pub const CONCEPT_REPLAY_DISTINCT_LATE_MAX: usize = 6;
    pub const CONCEPT_REPLAY_DOMINANT_LATE_MIN: usize = 2;
    pub const SURFACED_CONCEPT_REPLAY_COUNT_DRIFT_MAX: usize = 1;
    pub const SURFACED_CONCEPT_REPLAY_DISTINCT_LATE_MAX: usize = 4;
    pub const SURFACED_CONCEPT_REPLAY_DOMINANT_LATE_MIN: usize = 2;

    pub const RECALL_LATENCY_MAX_MS: f64 = 2.0;
}
