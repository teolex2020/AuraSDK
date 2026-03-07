use std::time::Instant;

use axum::{
    extract::{Json, Query, State},
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use metrics::{counter, gauge, histogram};
use rust_embed::RustEmbed;
use utoipa::OpenApi;

use crate::license;

use super::dto::*;
use super::state::ServerState;

#[derive(RustEmbed)]
#[folder = "ui/"]
struct Asset;

pub(super) async fn static_handler(uri: Uri) -> impl IntoResponse {
    let mut path = uri.path().trim_start_matches('/').to_string();

    if path.is_empty() {
        path = "index.html".to_string();
    }

    match Asset::get(&path) {
        Some(content) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            Response::builder()
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(content.data.into_owned()))
                .unwrap()
        }
        None => (StatusCode::NOT_FOUND, "404 Not Found").into_response(),
    }
}

pub(super) async fn prometheus_metrics(State(state): State<ServerState>) -> impl IntoResponse {
    let rendered = state.prom_handle.render();
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; version=0.0.4")], rendered).into_response()
}

#[utoipa::path(post, path = "/delete", request_body = DeleteRequest, responses((status = 200, description = "Memory deleted", body = DeleteResponse)))]
pub(super) async fn delete_memory(State(state): State<ServerState>, Json(payload): Json<DeleteRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let success = mem.delete_synapse(&payload.id);
    if success {
        counter!("aura_delete_total", "status" => "ok").increment(1);
        gauge!("aura_record_count").set(mem.count(None) as f64);
        (StatusCode::OK, Json(DeleteResponse { success: true }))
    } else {
        counter!("aura_delete_total", "status" => "not_found").increment(1);
        (StatusCode::NOT_FOUND, Json(DeleteResponse { success: false }))
    }
}

#[utoipa::path(post, path = "/update", request_body = UpdateRequest, responses((status = 200, description = "Memory updated", body = ProcessResponse)))]
pub(super) async fn update_memory(State(state): State<ServerState>, Json(payload): Json<UpdateRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    mem.delete_synapse(&payload.id);
    match mem.process(&payload.text, Some(true)) {
        Ok(_) => (StatusCode::OK, Json(ProcessResponse { status: "Updated".to_string() })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ProcessResponse { status: e.to_string() })),
    }
}

#[utoipa::path(get, path = "/health", responses((status = 200, description = "Health check", body = serde_json::Value)))]
pub(super) async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

#[utoipa::path(get, path = "/stats", responses((status = 200, description = "System statistics", body = StatsResponse)))]
pub(super) async fn stats(State(state): State<ServerState>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let count = mem.count(None);
    let license_info = license::get_license_info();
    let license_str = if license_info.hardware_bound { "Hardware Locked" } else { "Unlocked" };
    let phantoms = mem.phantom_count();

    gauge!("aura_record_count").set(count as f64);
    gauge!("aura_phantom_count").set(phantoms as f64);

    (StatusCode::OK, Json(StatsResponse {
        total_memories: count,
        license: license_str.to_string(),
        version: "v2.0".to_string(),
        phantom_count: phantoms,
    }))
}

#[utoipa::path(post, path = "/process", request_body = ProcessRequest, responses((status = 200, description = "Memory processed", body = ProcessResponse)))]
pub(super) async fn process(State(state): State<ServerState>, Json(payload): Json<ProcessRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let start = Instant::now();
    match mem.process(&payload.text, Some(payload.pin)) {
        Ok(status) => {
            let duration = start.elapsed().as_secs_f64();
            histogram!("aura_store_duration_seconds").record(duration);
            counter!("aura_store_total", "status" => "ok").increment(1);
            gauge!("aura_record_count").set(mem.count(None) as f64);
            (StatusCode::OK, Json(ProcessResponse { status }))
        }
        Err(e) => {
            counter!("aura_store_total", "status" => "error").increment(1);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ProcessResponse { status: e.to_string() }))
        }
    }
}

#[utoipa::path(post, path = "/retrieve", request_body = RetrieveRequest, responses((status = 200, description = "Retrieved memories", body = RetrieveResponse)))]
pub(super) async fn retrieve(State(state): State<ServerState>, Json(payload): Json<RetrieveRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let start = Instant::now();
    match mem.retrieve_full(&payload.query, payload.top_k) {
        Ok(records_with_scores) => {
            let duration = start.elapsed().as_secs_f64();
            histogram!("aura_recall_duration_seconds").record(duration);
            counter!("aura_recall_total", "status" => "ok").increment(1);
            gauge!("aura_recall_result_count").set(records_with_scores.len() as f64);
            let dtos: Vec<StoredRecordDTO> = records_with_scores.into_iter().map(|(r, tanimoto)| StoredRecordDTO {
                id: r.id,
                text: r.text,
                timestamp: r.timestamp,
                intensity: r.intensity,
                dna: r.dna,
                score: Some(tanimoto),
            }).collect();
            (StatusCode::OK, Json(RetrieveResponse { results: dtos }))
        }
        Err(_) => {
            counter!("aura_recall_total", "status" => "error").increment(1);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(RetrieveResponse { results: vec![] }))
        }
    }
}

#[utoipa::path(get, path = "/memories", params(("offset" = usize, Query, description = "Pagination offset"), ("limit" = usize, Query, description = "Page size (default 50)"), ("dna" = String, Query, description = "DNA filter: all, user_core, general, phantom")), responses((status = 200, description = "Paginated memory list", body = MemoriesResponse)))]
pub(super) async fn list_memories(State(state): State<ServerState>, Query(params): Query<MemoriesQuery>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let filter = if params.dna == "all" { None } else { Some(params.dna.as_str()) };
    let (records, total) = mem.list_memories(params.offset, params.limit, filter);
    let dtos: Vec<StoredRecordDTO> = records.into_iter().map(|r| StoredRecordDTO {
        id: r.id,
        text: r.text,
        timestamp: r.timestamp,
        intensity: r.intensity,
        dna: r.dna,
        score: None,
    }).collect();
    (StatusCode::OK, Json(MemoriesResponse { memories: dtos, total }))
}

#[utoipa::path(get, path = "/analytics", responses((status = 200, description = "Analytics data", body = AnalyticsResponse)))]
pub(super) async fn analytics(State(state): State<ServerState>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let (by_dna, total, oldest, newest) = mem.get_analytics();
    for (dna, count) in &by_dna {
        gauge!("aura_record_count_by_level", "level" => dna.clone()).set(*count as f64);
    }
    gauge!("aura_record_count").set(total as f64);
    (StatusCode::OK, Json(AnalyticsResponse { by_dna, total, oldest, newest }))
}

#[utoipa::path(post, path = "/batch-delete", request_body = BatchDeleteRequest, responses((status = 200, description = "Batch delete result", body = BatchDeleteResponse)))]
pub(super) async fn batch_delete(State(state): State<ServerState>, Json(payload): Json<BatchDeleteRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let deleted = mem.batch_delete(&payload.ids);
    counter!("aura_delete_total", "status" => "ok").increment(deleted as u64);
    gauge!("aura_record_count").set(mem.count(None) as f64);
    (StatusCode::OK, Json(BatchDeleteResponse { deleted }))
}

#[utoipa::path(post, path = "/ingest-batch", request_body = IngestBatchRequest, responses((status = 200, description = "Batch ingestion result", body = IngestBatchResponse)))]
pub(super) async fn ingest_batch(State(state): State<ServerState>, Json(payload): Json<IngestBatchRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let batch_size = payload.texts.len();
    let start = Instant::now();
    let result = if payload.pinned {
        mem.ingest_batch_pinned(payload.texts)
    } else {
        mem.ingest_batch(payload.texts)
    };
    match result {
        Ok(count) => {
            let duration = start.elapsed().as_secs_f64();
            histogram!("aura_batch_ingest_duration_seconds").record(duration);
            counter!("aura_store_total", "status" => "ok").increment(count as u64);
            gauge!("aura_record_count").set(mem.count(None) as f64);
            gauge!("aura_batch_size").set(batch_size as f64);
            (StatusCode::OK, Json(IngestBatchResponse { ingested: count, pinned: payload.pinned }))
        }
        Err(_) => {
            counter!("aura_store_total", "status" => "error").increment(batch_size as u64);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(IngestBatchResponse { ingested: 0, pinned: payload.pinned }))
        }
    }
}

#[utoipa::path(post, path = "/predict", request_body = PredictRequest, responses((status = 200, description = "Temporal prediction", body = PredictResponse)))]
pub(super) async fn predict(State(state): State<ServerState>, Json(payload): Json<PredictRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.retrieve_prediction(&payload.id) {
        Ok(Some(rec)) => (StatusCode::OK, Json(PredictResponse {
            found: true,
            result: Some(StoredRecordDTO {
                id: rec.id,
                text: rec.text,
                timestamp: rec.timestamp,
                intensity: rec.intensity,
                dna: rec.dna,
                score: None,
            }),
        })),
        Ok(None) => (StatusCode::OK, Json(PredictResponse { found: false, result: None })),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(PredictResponse { found: false, result: None })),
    }
}

#[utoipa::path(post, path = "/surprise", request_body = SurpriseRequest, responses((status = 200, description = "Anomaly detection score", body = SurpriseResponse)))]
pub(super) async fn surprise_handler(State(state): State<ServerState>, Json(payload): Json<SurpriseRequest>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.surprise(&payload.predicted_id, &payload.actual_text) {
        Ok(val) => (StatusCode::OK, Json(SurpriseResponse { surprise: val })),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(SurpriseResponse { surprise: -1.0 })),
    }
}

#[cfg(feature = "sync")]
pub(super) async fn export_sdr(State(state): State<ServerState>, Json(payload): Json<ExportSdrRequest>) -> impl IntoResponse {
    use crate::sync::SdrPrivacyConfig;

    let mem = state.memory.as_ref();
    let config = SdrPrivacyConfig {
        apply_noise: payload.apply_noise,
        drop_bits: payload.drop_bits,
        add_bits: payload.add_bits,
    };
    let fps = mem.export_sdr_fingerprints(payload.filter_dna.as_deref(), &config);
    let count = fps.len();
    let dtos: Vec<SdrFingerprintDTO> = fps.into_iter().map(|fp| SdrFingerprintDTO {
        id: fp.id,
        sdr_indices: fp.sdr_indices,
        timestamp: fp.timestamp,
        source_dna: fp.source_dna,
        intensity: fp.intensity,
        origin_node: fp.origin_node,
    }).collect();
    (StatusCode::OK, Json(ExportSdrResponse { fingerprints: dtos, count }))
}

#[cfg(feature = "sync")]
pub(super) async fn import_sdr(State(state): State<ServerState>, Json(payload): Json<ImportSdrRequest>) -> impl IntoResponse {
    use crate::sync::SdrFingerprint;

    let mem = state.memory.as_ref();
    let fps: Vec<SdrFingerprint> = payload.fingerprints.into_iter().map(|f| SdrFingerprint {
        id: f.id,
        sdr_indices: f.sdr_indices,
        timestamp: f.timestamp,
        source_dna: f.source_dna,
        intensity: f.intensity,
        origin_node: f.origin_node,
    }).collect();
    let imported = mem.import_sdr_fingerprints(fps);
    (StatusCode::OK, Json(ImportSdrResponse { imported }))
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Aura Memory API",
        version = "2.0.0",
        description = "Sub-millisecond deterministic memory for AI agents. SDR + inverted bitmap index."
    ),
    paths(
        health, process, retrieve, delete_memory, update_memory,
        stats, list_memories, analytics, batch_delete,
        ingest_batch, predict, surprise_handler
    ),
    components(schemas(
        ProcessRequest, ProcessResponse,
        RetrieveRequest, RetrieveResponse, StoredRecordDTO,
        DeleteRequest, DeleteResponse,
        UpdateRequest,
        StatsResponse,
        MemoriesQuery, MemoriesResponse,
        AnalyticsResponse,
        BatchDeleteRequest, BatchDeleteResponse,
        IngestBatchRequest, IngestBatchResponse,
        PredictRequest, PredictResponse,
        SurpriseRequest, SurpriseResponse,
    ))
)]
pub(super) struct ApiDoc;
