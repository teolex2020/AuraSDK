use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

#[derive(Deserialize, ToSchema)]
pub(super) struct ProcessRequest {
    pub(super) text: String,
    pub(super) pin: bool,
}

#[derive(Serialize, ToSchema)]
pub(super) struct ProcessResponse {
    pub(super) status: String,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct RetrieveRequest {
    pub(super) query: String,
    pub(super) top_k: usize,
}

#[derive(Serialize, ToSchema)]
pub(super) struct StoredRecordDTO {
    pub(super) id: String,
    pub(super) text: String,
    pub(super) timestamp: f64,
    pub(super) intensity: f32,
    pub(super) dna: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) score: Option<f32>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct RetrieveResponse {
    pub(super) results: Vec<StoredRecordDTO>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct DeleteRequest {
    pub(super) id: String,
}

#[derive(Serialize, ToSchema)]
pub(super) struct DeleteResponse {
    pub(super) success: bool,
}

#[derive(Serialize, ToSchema)]
pub(super) struct StatsResponse {
    pub(super) total_memories: usize,
    pub(super) license: String,
    pub(super) version: String,
    pub(super) phantom_count: usize,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct UpdateRequest {
    pub(super) id: String,
    pub(super) text: String,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct MemoriesQuery {
    #[serde(default)]
    pub(super) offset: usize,
    #[serde(default = "default_limit")]
    pub(super) limit: usize,
    #[serde(default = "default_dna")]
    pub(super) dna: String,
}

fn default_limit() -> usize {
    50
}
fn default_dna() -> String {
    "all".to_string()
}

#[derive(Serialize, ToSchema)]
pub(super) struct MemoriesResponse {
    pub(super) memories: Vec<StoredRecordDTO>,
    pub(super) total: usize,
}

#[derive(Serialize, ToSchema)]
pub(super) struct AnalyticsResponse {
    pub(super) by_dna: HashMap<String, usize>,
    pub(super) total: usize,
    pub(super) oldest: f64,
    pub(super) newest: f64,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct CrossNamespaceDigestQuery {
    /// Comma-separated namespaces to include. Omit for all namespaces.
    #[serde(default)]
    pub(super) namespaces: Option<String>,
    /// Maximum concepts returned per namespace. Clamped to 1..10.
    #[serde(default)]
    pub(super) top_concepts_limit: Option<usize>,
    /// Minimum record count required for a namespace to appear.
    #[serde(default)]
    pub(super) min_record_count: Option<usize>,
    /// Minimum similarity required for a pairwise entry to appear. Clamped to 0..1.
    #[serde(default)]
    pub(super) pairwise_similarity_threshold: Option<f32>,
    /// Comma-separated dimensions to include. Supported: concepts,tags,structural,causal,belief_states,corrections.
    #[serde(default)]
    pub(super) include_dimensions: Option<String>,
    /// When true, omit bulky per-namespace and per-pair lists while keeping summaries and scores.
    #[serde(default)]
    pub(super) compact_summary: Option<bool>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct ExplainRecordQuery {
    /// Target record ID to explain.
    pub(super) record_id: String,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct ExplainRecallQuery {
    /// Natural-language query to explain.
    pub(super) query: String,
    /// Maximum number of recall results to explain. Defaults to 10.
    #[serde(default)]
    pub(super) top_k: Option<usize>,
    /// Minimum record strength required for inclusion.
    #[serde(default)]
    pub(super) min_strength: Option<f32>,
    /// Whether graph/context expansion is enabled for the recall path.
    #[serde(default)]
    pub(super) expand_connections: Option<bool>,
    /// Optional comma-separated namespace filter.
    #[serde(default)]
    pub(super) namespaces: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct CorrectionLogQuery {
    /// Optional target kind filter: belief, causal_pattern, policy_hint, record.
    #[serde(default)]
    pub(super) target_kind: Option<String>,
    /// Optional target ID filter. Only applied together with target_kind.
    #[serde(default)]
    pub(super) target_id: Option<String>,
    /// Maximum entries to return, newest first. Defaults to 50.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct CorrectionReviewQueueQuery {
    /// Maximum review candidates returned, ordered by priority.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct ContradictionReviewQueueQuery {
    /// Optional namespace filter.
    #[serde(default)]
    pub(super) namespace: Option<String>,
    /// Maximum review candidates returned, ordered by priority.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct SuggestedCorrectionsQuery {
    /// Maximum suggested corrections returned, ordered by priority.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct NamespaceGovernanceQuery {
    /// Optional comma-separated namespaces to include. Omit for all namespaces.
    #[serde(default)]
    pub(super) namespaces: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct MemoryHealthQuery {
    /// Maximum top issues returned, clamped to a bounded range.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct BeliefInstabilityQuery {
    /// Maximum hotspots returned for high-volatility and low-stability lists.
    #[serde(default)]
    pub(super) limit: Option<usize>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct PolicyLifecycleQuery {
    /// Maximum grouped actions returned.
    #[serde(default)]
    pub(super) action_limit: Option<usize>,
    /// Maximum grouped domains returned.
    #[serde(default)]
    pub(super) domain_limit: Option<usize>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct ExplainRecordResponse {
    pub(super) found: bool,
    #[schema(value_type = Object)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) item: Option<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct ExplainRecallResponse {
    #[schema(value_type = Object)]
    pub(super) explanation: Value,
}

#[derive(Serialize, ToSchema)]
pub(super) struct ExplainabilityBundleResponse {
    pub(super) found: bool,
    #[schema(value_type = Object)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) bundle: Option<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct CorrectionLogResponse {
    pub(super) total: usize,
    #[schema(value_type = Vec<Object>)]
    pub(super) entries: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct CorrectionReviewQueueResponse {
    pub(super) total: usize,
    #[schema(value_type = Vec<Object>)]
    pub(super) entries: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct ContradictionReviewQueueResponse {
    pub(super) total: usize,
    #[schema(value_type = Vec<Object>)]
    pub(super) entries: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct SuggestedCorrectionsResponse {
    pub(super) scan_latency_ms: f64,
    pub(super) total: usize,
    #[schema(value_type = Vec<Object>)]
    pub(super) entries: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct NamespaceGovernanceResponse {
    pub(super) total: usize,
    #[schema(value_type = Vec<Object>)]
    pub(super) entries: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct MemoryHealthResponse {
    #[schema(value_type = Object)]
    pub(super) digest: Value,
}

#[derive(Serialize, ToSchema)]
pub(super) struct BeliefInstabilityResponse {
    #[schema(value_type = Object)]
    pub(super) summary: Value,
    #[schema(value_type = Vec<Object>)]
    pub(super) high_volatility: Vec<Value>,
    #[schema(value_type = Vec<Object>)]
    pub(super) low_stability: Vec<Value>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct PolicyLifecycleResponse {
    #[schema(value_type = Object)]
    pub(super) summary: Value,
    #[schema(value_type = Vec<Object>)]
    pub(super) suppressed_hints: Vec<Value>,
    #[schema(value_type = Vec<Object>)]
    pub(super) rejected_hints: Vec<Value>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct BatchDeleteRequest {
    pub(super) ids: Vec<String>,
}

#[derive(Serialize, ToSchema)]
pub(super) struct BatchDeleteResponse {
    pub(super) deleted: usize,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct IngestBatchRequest {
    pub(super) texts: Vec<String>,
    #[serde(default)]
    pub(super) pinned: bool,
}

#[derive(Serialize, ToSchema)]
pub(super) struct IngestBatchResponse {
    pub(super) ingested: usize,
    pub(super) pinned: bool,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct PredictRequest {
    pub(super) id: String,
}

#[derive(Serialize, ToSchema)]
pub(super) struct PredictResponse {
    pub(super) found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) result: Option<StoredRecordDTO>,
}

#[derive(Deserialize, ToSchema)]
pub(super) struct SurpriseRequest {
    pub(super) predicted_id: String,
    pub(super) actual_text: String,
}

#[derive(Serialize, ToSchema)]
pub(super) struct SurpriseResponse {
    pub(super) surprise: f32,
}

#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub(super) struct ExportSdrRequest {
    #[serde(default)]
    pub(super) filter_dna: Option<String>,
    #[serde(default)]
    pub(super) apply_noise: bool,
    #[serde(default)]
    pub(super) drop_bits: usize,
    #[serde(default)]
    pub(super) add_bits: usize,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub(super) struct ExportSdrResponse {
    pub(super) fingerprints: Vec<SdrFingerprintDTO>,
    pub(super) count: usize,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub(super) struct SdrFingerprintDTO {
    pub(super) id: String,
    pub(super) sdr_indices: Vec<u16>,
    pub(super) timestamp: f64,
    pub(super) source_dna: String,
    pub(super) intensity: f32,
    pub(super) origin_node: String,
}

#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub(super) struct ImportSdrRequest {
    pub(super) fingerprints: Vec<ImportSdrFingerprint>,
}

#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub(super) struct ImportSdrFingerprint {
    pub(super) id: String,
    pub(super) sdr_indices: Vec<u16>,
    pub(super) timestamp: f64,
    pub(super) source_dna: String,
    pub(super) intensity: f32,
    pub(super) origin_node: String,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub(super) struct ImportSdrResponse {
    pub(super) imported: usize,
}
