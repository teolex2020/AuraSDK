use std::collections::HashMap;

use serde::{Deserialize, Serialize};
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

fn default_limit() -> usize { 50 }
fn default_dna() -> String { "all".to_string() }

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
