use axum::{
    extract::{Json, Query, Request},
    routing::{get, post},
    Router,
    response::{IntoResponse, Response},
    http::{StatusCode, header, Uri},
    middleware::{self, Next},
};
use utoipa::OpenApi;
use utoipa::ToSchema;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::net::SocketAddr;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use crate::memory::AuraMemory;
use crate::license;
use lazy_static::lazy_static;
use rust_embed::RustEmbed;
use metrics::{counter, histogram};
use tower_http::cors::{CorsLayer, AllowOrigin};

// Global State
lazy_static! {
    static ref MEMORY: Arc<Mutex<Option<AuraMemory>>> = Arc::new(Mutex::new(None));
    static ref API_KEY: Mutex<Option<String>> = Mutex::new(None);
    static ref RATE_LIMIT_COUNTER: AtomicU64 = AtomicU64::new(0);
    static ref RATE_LIMIT_WINDOW_START: Mutex<Instant> = Mutex::new(Instant::now());
    static ref RATE_LIMIT_MAX: AtomicU64 = AtomicU64::new(100); // req/s, configurable
}

#[derive(RustEmbed)]
#[folder = "ui/"]
struct Asset;

// API endpoint paths that require authentication
const API_PATHS: &[&str] = &[
    "/process", "/retrieve", "/delete", "/update", "/stats",
    "/memories", "/analytics", "/batch-delete", "/ingest-batch",
    "/predict", "/surprise", "/export-sdr", "/import-sdr",
];

// --- Middleware: API Key Authentication ---
async fn auth_middleware(req: Request, next: Next) -> Response {
    let expected = {
        let guard = API_KEY.lock().unwrap();
        guard.clone()
    }; // guard dropped here, before any .await

    let expected = match expected {
        Some(k) => k,
        None => return next.run(req).await,
    };

    // Skip auth for health check and static assets
    let path = req.uri().path().to_string();
    let is_api = API_PATHS.iter().any(|p| path.starts_with(p));
    if !is_api {
        return next.run(req).await;
    }

    let auth_header = req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected {
                next.run(req).await
            } else {
                StatusCode::UNAUTHORIZED.into_response()
            }
        }
        _ => StatusCode::UNAUTHORIZED.into_response(),
    }
}

// --- Middleware: Rate Limiting (fixed window) ---
async fn rate_limit_middleware(req: Request, next: Next) -> Response {
    let max = RATE_LIMIT_MAX.load(Ordering::Relaxed);
    if max == 0 { return next.run(req).await; }

    let should_allow = {
        let mut window = RATE_LIMIT_WINDOW_START.lock().unwrap();
        let now = Instant::now();
        if now.duration_since(*window).as_secs() >= 1 {
            *window = now;
            RATE_LIMIT_COUNTER.store(1, Ordering::Relaxed);
            true
        } else {
            let count = RATE_LIMIT_COUNTER.fetch_add(1, Ordering::Relaxed);
            count < max
        }
    }; // guard dropped here

    if should_allow {
        next.run(req).await
    } else {
        StatusCode::TOO_MANY_REQUESTS.into_response()
    }
}

// --- Prometheus metrics ---
lazy_static! {
    static ref PROM_HANDLE: Mutex<Option<metrics_exporter_prometheus::PrometheusHandle>> = Mutex::new(None);
}

// Middleware: track request count + latency
async fn metrics_middleware(req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();
    let start = Instant::now();

    let response = next.run(req).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    counter!("aura_http_requests_total", "method" => method.clone(), "path" => path.clone(), "status" => status).increment(1);
    histogram!("aura_http_request_duration_seconds", "method" => method, "path" => path).record(duration);

    response
}

// GET /metrics — Prometheus scrape endpoint
async fn prometheus_metrics() -> impl IntoResponse {
    let guard = PROM_HANDLE.lock().unwrap();
    match guard.as_ref() {
        Some(handle) => {
            let rendered = handle.render();
            (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; version=0.0.4")], rendered).into_response()
        }
        None => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

// Request/Response Types (Process, Retrieve, etc.)
#[derive(Deserialize, ToSchema)]
pub struct ProcessRequest {
    text: String,
    pin: bool,
}

#[derive(Serialize, ToSchema)]
pub struct ProcessResponse {
    status: String,
}

#[derive(Deserialize, ToSchema)]
pub struct RetrieveRequest {
    query: String,
    top_k: usize,
}

#[derive(Serialize, ToSchema)]
pub struct StoredRecordDTO {
    id: String,
    text: String,
    timestamp: f64,
    intensity: f32,
    dna: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
}

#[derive(Serialize, ToSchema)]
pub struct RetrieveResponse {
    results: Vec<StoredRecordDTO>,
}

#[derive(Deserialize, ToSchema)]
pub struct DeleteRequest {
    id: String,
}

#[derive(Serialize, ToSchema)]
pub struct DeleteResponse {
    success: bool,
}

#[derive(Serialize, ToSchema)]
pub struct StatsResponse {
    total_memories: usize,
    license: String,
    version: String,
    plasticity_boosts: u64,
    plasticity_decays: u64,
    plasticity_immune: u64,
    phantom_count: usize,
}

#[derive(Deserialize, ToSchema)]
pub struct UpdateRequest {
    id: String,
    text: String,
}

// New: Memories list query params
#[derive(Deserialize, ToSchema)]
pub struct MemoriesQuery {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default = "default_dna")]
    dna: String,
}

fn default_limit() -> usize { 50 }
fn default_dna() -> String { "all".to_string() }

#[derive(Serialize, ToSchema)]
pub struct MemoriesResponse {
    memories: Vec<StoredRecordDTO>,
    total: usize,
}

// New: Analytics response
#[derive(Serialize, ToSchema)]
pub struct AnalyticsResponse {
    by_dna: HashMap<String, usize>,
    total: usize,
    oldest: f64,
    newest: f64,
}

// New: Batch delete
#[derive(Deserialize, ToSchema)]
pub struct BatchDeleteRequest {
    ids: Vec<String>,
}

#[derive(Serialize, ToSchema)]
pub struct BatchDeleteResponse {
    deleted: usize,
}

// New: Batch ingest
#[derive(Deserialize, ToSchema)]
pub struct IngestBatchRequest {
    texts: Vec<String>,
    #[serde(default)]
    pinned: bool,
}

#[derive(Serialize, ToSchema)]
pub struct IngestBatchResponse {
    ingested: usize,
    pinned: bool,
}

// New: Prediction
#[derive(Deserialize, ToSchema)]
pub struct PredictRequest {
    id: String,
}

#[derive(Serialize, ToSchema)]
pub struct PredictResponse {
    found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<StoredRecordDTO>,
}

// New: Surprise
#[derive(Deserialize, ToSchema)]
pub struct SurpriseRequest {
    predicted_id: String,
    actual_text: String,
}

#[derive(Serialize, ToSchema)]
pub struct SurpriseResponse {
    surprise: f32,
}

// SDR Exchange types
#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub struct ExportSdrRequest {
    #[serde(default)]
    filter_dna: Option<String>,
    #[serde(default)]
    apply_noise: bool,
    #[serde(default)]
    drop_bits: usize,
    #[serde(default)]
    add_bits: usize,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub struct ExportSdrResponse {
    fingerprints: Vec<SdrFingerprintDTO>,
    count: usize,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub struct SdrFingerprintDTO {
    id: String,
    sdr_indices: Vec<u16>,
    timestamp: f64,
    source_dna: String,
    intensity: f32,
    origin_node: String,
}

#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub struct ImportSdrRequest {
    fingerprints: Vec<ImportSdrFingerprint>,
}

#[cfg(feature = "sync")]
#[derive(Deserialize)]
pub struct ImportSdrFingerprint {
    id: String,
    sdr_indices: Vec<u16>,
    timestamp: f64,
    source_dna: String,
    intensity: f32,
    origin_node: String,
}

#[cfg(feature = "sync")]
#[derive(Serialize)]
pub struct ImportSdrResponse {
    imported: usize,
}

// HANDLERS

// 1. Static File Handler (Embedded UI)
async fn static_handler(uri: Uri) -> impl IntoResponse {
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
        },
        None => {
            (StatusCode::NOT_FOUND, "404 Not Found").into_response()
        }
    }
}

// 2. API Handlers
#[utoipa::path(post, path = "/delete", request_body = DeleteRequest, responses((status = 200, description = "Memory deleted", body = DeleteResponse)))]
async fn delete_memory(Json(payload): Json<DeleteRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        let success = mem.delete_synapse(&payload.id);
        if success {
             (StatusCode::OK, Json(DeleteResponse { success: true }))
        } else {
             (StatusCode::NOT_FOUND, Json(DeleteResponse { success: false }))
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(DeleteResponse { success: false }))
    }
}

#[utoipa::path(post, path = "/update", request_body = UpdateRequest, responses((status = 200, description = "Memory updated", body = ProcessResponse)))]
async fn update_memory(Json(payload): Json<UpdateRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        mem.delete_synapse(&payload.id);
        match mem.process(&payload.text, true) {
             Ok(_) => (StatusCode::OK, Json(ProcessResponse { status: "Updated".to_string() })),
             Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ProcessResponse { status: e.to_string() })),
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ProcessResponse { status: "Memory not initialized".to_string() }))
    }
}

#[utoipa::path(get, path = "/health", responses((status = 200, description = "Health check", body = serde_json::Value)))]
async fn health() -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if mutex.is_some() {
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"status": "not initialized"})))
    }
}

#[utoipa::path(get, path = "/stats", responses((status = 200, description = "System statistics", body = StatsResponse)))]
async fn stats() -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        let count = mem.count();
        let license_info = license::get_license_info();
        let license_str = if license_info.hardware_bound { "Hardware Locked" } else { "Unlocked" };
        let (boosts, decays, immune) = mem.plasticity_stats();
        let phantoms = mem.phantom_count();

        (StatusCode::OK, Json(StatsResponse {
            total_memories: count,
            license: license_str.to_string(),
            version: "v2.0".to_string(),
            plasticity_boosts: boosts,
            plasticity_decays: decays,
            plasticity_immune: immune,
            phantom_count: phantoms,
        }))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(StatsResponse {
            total_memories: 0,
            license: "Unknown".to_string(),
            version: "v2.0".to_string(),
            plasticity_boosts: 0,
            plasticity_decays: 0,
            plasticity_immune: 0,
            phantom_count: 0,
        }))
    }
}

#[utoipa::path(post, path = "/process", request_body = ProcessRequest, responses((status = 200, description = "Memory processed", body = ProcessResponse)))]
async fn process(Json(payload): Json<ProcessRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        match mem.process(&payload.text, payload.pin) {
            Ok(status) => (StatusCode::OK, Json(ProcessResponse { status })),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ProcessResponse { status: e.to_string() })),
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ProcessResponse { status: "Memory not initialized".into() }))
    }
}

#[utoipa::path(post, path = "/retrieve", request_body = RetrieveRequest, responses((status = 200, description = "Retrieved memories", body = RetrieveResponse)))]
async fn retrieve(Json(payload): Json<RetrieveRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        match mem.retrieve_full(&payload.query, payload.top_k) {
            Ok(records_with_scores) => {
                let dtos: Vec<StoredRecordDTO> = records_with_scores.into_iter().map(|(r, tanimoto)| StoredRecordDTO {
                    id: r.id,
                    text: r.text,
                    timestamp: r.timestamp,
                    intensity: r.intensity,
                    dna: r.dna,
                    score: Some(tanimoto),
                }).collect();
                (StatusCode::OK, Json(RetrieveResponse { results: dtos }))
            },
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(RetrieveResponse { results: vec![] }))
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(RetrieveResponse { results: vec![] }))
    }
}

// 3. New Handlers

#[utoipa::path(get, path = "/memories", params(("offset" = usize, Query, description = "Pagination offset"), ("limit" = usize, Query, description = "Page size (default 50)"), ("dna" = String, Query, description = "DNA filter: all, user_core, general, phantom")), responses((status = 200, description = "Paginated memory list", body = MemoriesResponse)))]
async fn list_memories(Query(params): Query<MemoriesQuery>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
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
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(MemoriesResponse { memories: vec![], total: 0 }))
    }
}

#[utoipa::path(get, path = "/analytics", responses((status = 200, description = "Analytics data", body = AnalyticsResponse)))]
async fn analytics() -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        let (by_dna, total, oldest, newest) = mem.get_analytics();
        (StatusCode::OK, Json(AnalyticsResponse { by_dna, total, oldest, newest }))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(AnalyticsResponse {
            by_dna: HashMap::new(), total: 0, oldest: 0.0, newest: 0.0,
        }))
    }
}

#[utoipa::path(post, path = "/batch-delete", request_body = BatchDeleteRequest, responses((status = 200, description = "Batch delete result", body = BatchDeleteResponse)))]
async fn batch_delete(Json(payload): Json<BatchDeleteRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        let deleted = mem.batch_delete(&payload.ids);
        (StatusCode::OK, Json(BatchDeleteResponse { deleted }))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(BatchDeleteResponse { deleted: 0 }))
    }
}

// 4. Batch Ingest Handler
#[utoipa::path(post, path = "/ingest-batch", request_body = IngestBatchRequest, responses((status = 200, description = "Batch ingestion result", body = IngestBatchResponse)))]
async fn ingest_batch(Json(payload): Json<IngestBatchRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        let result = if payload.pinned {
            mem.ingest_batch_pinned(payload.texts)
        } else {
            mem.ingest_batch(payload.texts)
        };
        match result {
            Ok(count) => (StatusCode::OK, Json(IngestBatchResponse { ingested: count, pinned: payload.pinned })),
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(IngestBatchResponse { ingested: 0, pinned: payload.pinned })),
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(IngestBatchResponse { ingested: 0, pinned: false }))
    }
}

// 5. Prediction Handler
#[utoipa::path(post, path = "/predict", request_body = PredictRequest, responses((status = 200, description = "Temporal prediction", body = PredictResponse)))]
async fn predict(Json(payload): Json<PredictRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
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
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(PredictResponse { found: false, result: None }))
    }
}

// 6. Surprise Handler
#[utoipa::path(post, path = "/surprise", request_body = SurpriseRequest, responses((status = 200, description = "Anomaly detection score", body = SurpriseResponse)))]
async fn surprise_handler(Json(payload): Json<SurpriseRequest>) -> impl IntoResponse {
    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
        match mem.surprise(&payload.predicted_id, &payload.actual_text) {
            Ok(val) => (StatusCode::OK, Json(SurpriseResponse { surprise: val })),
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(SurpriseResponse { surprise: -1.0 })),
        }
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(SurpriseResponse { surprise: -1.0 }))
    }
}

// 7. SDR Exchange Handlers
#[cfg(feature = "sync")]
async fn export_sdr(Json(payload): Json<ExportSdrRequest>) -> impl IntoResponse {
    use crate::sync::SdrPrivacyConfig;

    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
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
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ExportSdrResponse { fingerprints: vec![], count: 0 }))
    }
}

#[cfg(feature = "sync")]
async fn import_sdr(Json(payload): Json<ImportSdrRequest>) -> impl IntoResponse {
    use crate::sync::SdrFingerprint;

    let mutex = MEMORY.lock().unwrap();
    if let Some(mem) = mutex.as_ref() {
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
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ImportSdrResponse { imported: 0 }))
    }
}

// --- OpenAPI Spec ---
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
struct ApiDoc;

// Start Server
pub fn start_server(port: u16, storage_path: &str) -> anyhow::Result<()> {
    // Init Memory
    let mem = AuraMemory::new(storage_path)?;
    {
        let mut global = MEMORY.lock().unwrap();
        *global = Some(mem);
    }

    // ENV: AURA_API_KEY — if set, all API endpoints require Bearer token
    if let Ok(key) = std::env::var("AURA_API_KEY") {
        if !key.is_empty() {
            let mut k = API_KEY.lock().unwrap();
            *k = Some(key);
            println!("[AUTH] API Key authentication enabled");
        }
    }

    // ENV: AURA_RATE_LIMIT — max requests per second (0 = disabled)
    if let Ok(val) = std::env::var("AURA_RATE_LIMIT") {
        if let Ok(n) = val.parse::<u64>() {
            RATE_LIMIT_MAX.store(n, Ordering::Relaxed);
            if n > 0 {
                println!("[RATE] Rate limit: {} req/s", n);
            } else {
                println!("[RATE] Rate limiting disabled");
            }
        }
    }

    // ENV: AURA_BIND — bind address (default: 127.0.0.1:<port>)
    let bind_addr: SocketAddr = std::env::var("AURA_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], port)));

    // ENV: AURA_CORS_ORIGINS — comma-separated allowed origins (default: same-origin only)
    // Examples: AURA_CORS_ORIGINS=* (allow all), AURA_CORS_ORIGINS=https://app.example.com,https://admin.example.com
    let cors_layer = match std::env::var("AURA_CORS_ORIGINS").ok() {
        Some(val) if val == "*" => {
            println!("[CORS] Allowing all origins (development mode)");
            CorsLayer::permissive()
        }
        Some(val) if !val.is_empty() => {
            let origins: Vec<axum::http::HeaderValue> = val.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let count = origins.len();
            println!("[CORS] Allowed origins: {} configured", count);
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        }
        _ => {
            println!("[CORS] Same-origin only (set AURA_CORS_ORIGINS to configure)");
            CorsLayer::new()
                .allow_origin(AllowOrigin::exact("null".parse().unwrap()))
                .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        }
    };

    // Prometheus metrics recorder (global)
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder();
    let prom_handle = recorder.handle();
    metrics::set_global_recorder(recorder).ok();
    {
        let mut guard = PROM_HANDLE.lock().unwrap();
        *guard = Some(prom_handle);
    }
    println!("[METRICS] Prometheus metrics enabled at /metrics");

    // Structured logging: RUST_LOG env filter, JSON if AURA_LOG_JSON=1
    let use_json = std::env::var("AURA_LOG_JSON").ok().map(|v| v == "1").unwrap_or(false);
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if use_json {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .try_init();
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // API routes (protected by auth + rate limiting + metrics)
        let api = Router::new()
            .route("/health", get(health))
            .route("/process", post(process))
            .route("/retrieve", post(retrieve))
            .route("/delete", post(delete_memory))
            .route("/update", post(update_memory))
            .route("/stats", get(stats))
            .route("/memories", get(list_memories))
            .route("/analytics", get(analytics))
            .route("/batch-delete", post(batch_delete))
            .route("/ingest-batch", post(ingest_batch))
            .route("/predict", post(predict))
            .route("/surprise", post(surprise_handler));

        #[cfg(feature = "sync")]
        let api = api
            .route("/export-sdr", post(export_sdr))
            .route("/import-sdr", post(import_sdr));

        let api = api
            .layer(middleware::from_fn(rate_limit_middleware))
            .layer(middleware::from_fn(auth_middleware));

        // Swagger UI + Prometheus metrics endpoint (outside auth)
        let app = api
            .merge(
                utoipa_swagger_ui::SwaggerUi::new("/docs")
                    .url("/openapi.json", ApiDoc::openapi())
            )
            .route("/metrics", get(prometheus_metrics))
            .route("/", get(static_handler))
            .route("/*file", get(static_handler))
            .layer(middleware::from_fn(metrics_middleware))
            .layer(cors_layer);

        // TLS configuration: AURA_TLS_CERT + AURA_TLS_KEY
        let tls_cert = std::env::var("AURA_TLS_CERT").ok();
        let tls_key = std::env::var("AURA_TLS_KEY").ok();
        let use_tls = tls_cert.is_some() && tls_key.is_some();

        let scheme = if use_tls { "https" } else { "http" };
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║           AURA DASHBOARD v2.0 RRF Fusion                ║");
        println!("║           Open: {}://{}                  ║", scheme, bind_addr);
        println!("╚══════════════════════════════════════════════════════════╝");

        // Open browser (only for localhost, non-TLS)
        if bind_addr.ip().is_loopback() && !use_tls {
            if let Err(_e) = open::that(format!("http://{}", bind_addr)) {
                // ignore error
            }
        }

        if use_tls {
            // HTTPS mode: axum-server with rustls
            let cert_path = tls_cert.unwrap();
            let key_path = tls_key.unwrap();
            println!("[TLS] Loading cert: {}, key: {}", cert_path, key_path);

            let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                .await
                .expect("Failed to load TLS cert/key. Check AURA_TLS_CERT and AURA_TLS_KEY paths.");

            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();

            // Spawn shutdown signal listener
            tokio::spawn(async move {
                shutdown_signal().await;
                shutdown_handle.graceful_shutdown(Some(std::time::Duration::from_secs(10)));
            });

            println!("[TLS] HTTPS enabled");
            axum_server::bind_rustls(bind_addr, tls_config)
                .handle(handle)
                .serve(app.into_make_service())
                .await
                .unwrap();
        } else {
            // HTTP mode: standard axum serve
            let listener = tokio::net::TcpListener::bind(bind_addr).await.unwrap();
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .unwrap();
        }

        // Post-shutdown: flush all pending data
        println!("\n[SHUTDOWN] Flushing memory to disk...");
        let mutex = MEMORY.lock().unwrap();
        if let Some(mem) = mutex.as_ref() {
            mem.flush_consolidation();
            let _ = mem.flush();
            println!("[SHUTDOWN] All data saved. Goodbye.");
        }
    });

    Ok(())
}

// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { println!("\n[SHUTDOWN] Received Ctrl+C, shutting down gracefully..."); },
        _ = terminate => { println!("\n[SHUTDOWN] Received SIGTERM, shutting down gracefully..."); },
    }
}

// Temporary stubs for missing dependencies or methods
mod open {
    pub fn that<T: AsRef<std::ffi::OsStr>>(path: T) -> std::io::Result<()> {
        // Simple Windows fallback
        std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path.as_ref())
            .spawn()?;
        Ok(())
    }
}
