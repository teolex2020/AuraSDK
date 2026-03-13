use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::aura::Aura;

pub(super) struct RateLimitState {
    pub(super) counter: AtomicU64,
    pub(super) window_start: Mutex<Instant>,
    pub(super) max: AtomicU64,
}

#[derive(Clone)]
pub(super) struct ServerState {
    pub(super) memory: Arc<Aura>,
    pub(super) api_key: Option<Arc<str>>,
    pub(super) rate_limit: Arc<RateLimitState>,
    pub(super) prom_handle: Arc<metrics_exporter_prometheus::PrometheusHandle>,
}

pub(super) const API_PATHS: &[&str] = &[
    "/process",
    "/retrieve",
    "/delete",
    "/update",
    "/stats",
    "/memories",
    "/analytics",
    "/batch-delete",
    "/ingest-batch",
    "/predict",
    "/surprise",
    "/export-sdr",
    "/import-sdr",
];
