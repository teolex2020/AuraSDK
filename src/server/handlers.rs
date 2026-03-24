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
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        rendered,
    )
        .into_response()
}

#[utoipa::path(post, path = "/delete", request_body = DeleteRequest, responses((status = 200, description = "Memory deleted", body = DeleteResponse)))]
pub(super) async fn delete_memory(
    State(state): State<ServerState>,
    Json(payload): Json<DeleteRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let success = mem.delete_synapse(&payload.id);
    if success {
        counter!("aura_delete_total", "status" => "ok").increment(1);
        gauge!("aura_record_count").set(mem.count(None) as f64);
        (StatusCode::OK, Json(DeleteResponse { success: true }))
    } else {
        counter!("aura_delete_total", "status" => "not_found").increment(1);
        (
            StatusCode::NOT_FOUND,
            Json(DeleteResponse { success: false }),
        )
    }
}

#[utoipa::path(post, path = "/update", request_body = UpdateRequest, responses((status = 200, description = "Memory updated", body = ProcessResponse)))]
pub(super) async fn update_memory(
    State(state): State<ServerState>,
    Json(payload): Json<UpdateRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    mem.delete_synapse(&payload.id);
    match mem.process(&payload.text, Some(true)) {
        Ok(_) => (
            StatusCode::OK,
            Json(ProcessResponse {
                status: "Updated".to_string(),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ProcessResponse {
                status: e.to_string(),
            }),
        ),
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
    let license_str = if license_info.hardware_bound {
        "Hardware Locked"
    } else {
        "Unlocked"
    };
    let phantoms = mem.phantom_count();

    gauge!("aura_record_count").set(count as f64);
    gauge!("aura_phantom_count").set(phantoms as f64);

    (
        StatusCode::OK,
        Json(StatsResponse {
            total_memories: count,
            license: license_str.to_string(),
            version: "v2.0".to_string(),
            phantom_count: phantoms,
        }),
    )
}

#[utoipa::path(post, path = "/process", request_body = ProcessRequest, responses((status = 200, description = "Memory processed", body = ProcessResponse)))]
pub(super) async fn process(
    State(state): State<ServerState>,
    Json(payload): Json<ProcessRequest>,
) -> impl IntoResponse {
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
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ProcessResponse {
                    status: e.to_string(),
                }),
            )
        }
    }
}

#[utoipa::path(post, path = "/retrieve", request_body = RetrieveRequest, responses((status = 200, description = "Retrieved memories", body = RetrieveResponse)))]
pub(super) async fn retrieve(
    State(state): State<ServerState>,
    Json(payload): Json<RetrieveRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let start = Instant::now();
    match mem.retrieve_full(&payload.query, payload.top_k) {
        Ok(records_with_scores) => {
            let duration = start.elapsed().as_secs_f64();
            histogram!("aura_recall_duration_seconds").record(duration);
            counter!("aura_recall_total", "status" => "ok").increment(1);
            gauge!("aura_recall_result_count").set(records_with_scores.len() as f64);
            let dtos: Vec<StoredRecordDTO> = records_with_scores
                .into_iter()
                .map(|(r, tanimoto)| StoredRecordDTO {
                    id: r.id,
                    text: r.text,
                    timestamp: r.timestamp,
                    intensity: r.intensity,
                    dna: r.dna,
                    score: Some(tanimoto),
                })
                .collect();
            (StatusCode::OK, Json(RetrieveResponse { results: dtos }))
        }
        Err(_) => {
            counter!("aura_recall_total", "status" => "error").increment(1);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(RetrieveResponse { results: vec![] }),
            )
        }
    }
}

#[utoipa::path(get, path = "/memories", params(("offset" = usize, Query, description = "Pagination offset"), ("limit" = usize, Query, description = "Page size (default 50)"), ("dna" = String, Query, description = "DNA filter: all, user_core, general, phantom")), responses((status = 200, description = "Paginated memory list", body = MemoriesResponse)))]
pub(super) async fn list_memories(
    State(state): State<ServerState>,
    Query(params): Query<MemoriesQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let filter = if params.dna == "all" {
        None
    } else {
        Some(params.dna.as_str())
    };
    let (records, total) = mem.list_memories(params.offset, params.limit, filter);
    let dtos: Vec<StoredRecordDTO> = records
        .into_iter()
        .map(|r| StoredRecordDTO {
            id: r.id,
            text: r.text,
            timestamp: r.timestamp,
            intensity: r.intensity,
            dna: r.dna,
            score: None,
        })
        .collect();
    (
        StatusCode::OK,
        Json(MemoriesResponse {
            memories: dtos,
            total,
        }),
    )
}

#[utoipa::path(get, path = "/analytics", responses((status = 200, description = "Analytics data", body = AnalyticsResponse)))]
pub(super) async fn analytics(State(state): State<ServerState>) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let (by_dna, total, oldest, newest) = mem.get_analytics();
    for (dna, count) in &by_dna {
        gauge!("aura_record_count_by_level", "level" => dna.clone()).set(*count as f64);
    }
    gauge!("aura_record_count").set(total as f64);
    (
        StatusCode::OK,
        Json(AnalyticsResponse {
            by_dna,
            total,
            oldest,
            newest,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/memory-health",
    params(
        ("limit" = Option<usize>, Query, description = "Maximum top issues returned; defaults to 8")
    ),
    responses(
        (status = 200, description = "Compact operator-facing memory health digest", body = MemoryHealthResponse, example = json!({
            "digest": {
                "total_records": 42,
                "startup_has_recovery_warnings": false,
                "high_volatility_belief_count": 3,
                "recent_correction_count": 2,
                "maintenance_trend_direction": "stable",
                "top_issues": [{"kind": "policy_pressure", "severity": "high"}]
            }
        }))
    )
)]
pub(super) async fn memory_health(
    State(state): State<ServerState>,
    Query(query): Query<MemoryHealthQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let digest = mem.get_memory_health_digest(query.limit);
    (
        StatusCode::OK,
        Json(MemoryHealthResponse {
            digest: serde_json::to_value(digest).unwrap_or(serde_json::Value::Null),
        }),
    )
}

#[utoipa::path(
    get,
    path = "/belief-instability",
    params(
        ("limit" = Option<usize>, Query, description = "Maximum hotspots returned for each instability list; defaults to 8")
    ),
    responses(
        (status = 200, description = "Belief instability summary with hotspot lists", body = BeliefInstabilityResponse, example = json!({
            "summary": {
                "total_beliefs": 12,
                "high_volatility_count": 2,
                "low_stability_count": 1
            },
            "high_volatility": [{"belief_id": "belief-1", "volatility": 0.81}],
            "low_stability": [{"belief_id": "belief-2", "stability": 0.22}]
        }))
    )
)]
pub(super) async fn belief_instability(
    State(state): State<ServerState>,
    Query(query): Query<BeliefInstabilityQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let limit = query.limit.unwrap_or(8).clamp(1, 32);
    let summary = serde_json::to_value(mem.get_belief_instability_summary())
        .unwrap_or(serde_json::Value::Null);
    let high_volatility = mem
        .get_high_volatility_beliefs(None, Some(limit))
        .into_iter()
        .filter_map(|item| serde_json::to_value(item).ok())
        .collect::<Vec<_>>();
    let low_stability = mem
        .get_low_stability_beliefs(None, Some(limit))
        .into_iter()
        .filter_map(|item| serde_json::to_value(item).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(BeliefInstabilityResponse {
            summary,
            high_volatility,
            low_stability,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/policy-lifecycle",
    params(
        ("action_limit" = Option<usize>, Query, description = "Maximum grouped actions returned"),
        ("domain_limit" = Option<usize>, Query, description = "Maximum grouped domains returned")
    ),
    responses(
        (status = 200, description = "Policy lifecycle summary with suppressed and rejected hints", body = PolicyLifecycleResponse, example = json!({
            "summary": {
                "total_hints": 9,
                "suppressed_count": 2,
                "rejected_count": 1
            },
            "suppressed_hints": [{"id": "policy-1"}],
            "rejected_hints": [{"id": "policy-2"}]
        }))
    )
)]
pub(super) async fn policy_lifecycle(
    State(state): State<ServerState>,
    Query(query): Query<PolicyLifecycleQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let summary = serde_json::to_value(
        mem.get_policy_lifecycle_summary(query.action_limit, query.domain_limit),
    )
    .unwrap_or(serde_json::Value::Null);
    let limit = query.action_limit.unwrap_or(8).clamp(1, 32);
    let suppressed_hints = mem
        .get_suppressed_policy_hints(None, Some(limit))
        .into_iter()
        .filter_map(|item| serde_json::to_value(item).ok())
        .collect::<Vec<_>>();
    let rejected_hints = mem
        .get_rejected_policy_hints(None, Some(limit))
        .into_iter()
        .filter_map(|item| serde_json::to_value(item).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(PolicyLifecycleResponse {
            summary,
            suppressed_hints,
            rejected_hints,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/explain-record",
    params(
        ("record_id" = String, Query, description = "Record ID to explain", example = "rec_123")
    ),
    responses(
        (status = 200, description = "Explainability payload for one record", body = ExplainRecordResponse, example = json!({
            "found": true,
            "item": {
                "record_id": "rec_123",
                "namespace": "default",
                "content_preview": "deploy rollback stabilized checkout",
                "belief": {"id": "belief_1", "state": "resolved", "confidence": 0.82},
                "concepts": [],
                "causal_patterns": [],
                "policy_hints": []
            }
        })),
        (status = 404, description = "Record not found", body = ExplainRecordResponse)
    )
)]
pub(super) async fn explain_record(
    State(state): State<ServerState>,
    Query(query): Query<ExplainRecordQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.explain_record(&query.record_id) {
        Some(item) => (
            StatusCode::OK,
            Json(ExplainRecordResponse {
                found: true,
                item: serde_json::to_value(item).ok(),
            }),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(ExplainRecordResponse {
                found: false,
                item: None,
            }),
        ),
    }
}

#[utoipa::path(
    get,
    path = "/explainability-bundle",
    params(
        ("record_id" = String, Query, description = "Record ID to inspect as a single bounded explainability bundle", example = "rec_123")
    ),
    responses(
        (status = 200, description = "Bounded explainability bundle for one record", body = ExplainabilityBundleResponse, example = json!({
            "found": true,
            "bundle": {
                "record_id": "rec_123",
                "explanation": {"record_id": "rec_123"},
                "provenance": {"record_id": "rec_123", "steps": ["record -> belief"]},
                "record_corrections": [],
                "belief_corrections": [],
                "causal_corrections": [],
                "policy_corrections": []
            }
        })),
        (status = 404, description = "Record not found", body = ExplainabilityBundleResponse)
    )
)]
pub(super) async fn explainability_bundle(
    State(state): State<ServerState>,
    Query(query): Query<ExplainRecordQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.explainability_bundle(&query.record_id) {
        Some(bundle) => (
            StatusCode::OK,
            Json(ExplainabilityBundleResponse {
                found: true,
                bundle: serde_json::to_value(bundle).ok(),
            }),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(ExplainabilityBundleResponse {
                found: false,
                bundle: None,
            }),
        ),
    }
}

#[utoipa::path(
    get,
    path = "/explain-recall",
    params(
        ("query" = String, Query, description = "Natural-language query to explain", example = "deploy stability rollback"),
        ("top_k" = Option<usize>, Query, description = "Maximum explained recall results; defaults to 10"),
        ("min_strength" = Option<f32>, Query, description = "Minimum record strength required for inclusion"),
        ("expand_connections" = Option<bool>, Query, description = "Whether graph/context expansion is enabled"),
        ("namespaces" = Option<String>, Query, description = "Comma-separated namespace filter")
    ),
    responses(
        (status = 200, description = "Full recall explanation with bounded trace scoring", body = ExplainRecallResponse, example = json!({
            "explanation": {
                "query": "deploy stability rollback",
                "top_k": 5,
                "result_count": 1,
                "belief_rerank_mode": "limited",
                "items": [{"record_id": "rec_123", "rank": 1, "score": 0.91}]
            }
        }))
    )
)]
pub(super) async fn explain_recall(
    State(state): State<ServerState>,
    Query(query): Query<ExplainRecallQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let namespaces_owned = query.namespaces.as_ref().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|namespace| !namespace.is_empty())
            .collect::<Vec<_>>()
    });
    let explanation = mem.explain_recall(
        &query.query,
        query.top_k.or(Some(10)),
        query.min_strength,
        query.expand_connections,
        namespaces_owned.as_deref(),
    );
    (
        StatusCode::OK,
        Json(ExplainRecallResponse {
            explanation: serde_json::to_value(explanation).unwrap_or(serde_json::Value::Null),
        }),
    )
}

#[utoipa::path(
    get,
    path = "/correction-log",
    params(
        ("target_kind" = Option<String>, Query, description = "Optional target kind filter: belief, causal_pattern, policy_hint, record"),
        ("target_id" = Option<String>, Query, description = "Optional target ID filter; only applied together with target_kind"),
        ("limit" = Option<usize>, Query, description = "Maximum entries returned, newest first; defaults to 50")
    ),
    responses(
        (status = 200, description = "Correction log entries for all targets or a specific target", body = CorrectionLogResponse, example = json!({
            "total": 1,
            "entries": [{
                "target_kind": "policy_hint",
                "target_id": "policy-1",
                "operation": "retract",
                "reason": "superseded_runbook"
            }]
        }))
    )
)]
pub(super) async fn correction_log(
    State(state): State<ServerState>,
    Query(query): Query<CorrectionLogQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let max = query.limit.unwrap_or(50).min(200);
    let mut entries = if let (Some(target_kind), Some(target_id)) =
        (query.target_kind.as_deref(), query.target_id.as_deref())
    {
        mem.get_correction_log_for_target(target_kind, target_id)
    } else {
        mem.get_correction_log()
    };
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries.truncate(max);
    let entries = entries
        .into_iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(CorrectionLogResponse {
            total: entries.len(),
            entries,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/correction-review-queue",
    params(
        ("limit" = Option<usize>, Query, description = "Maximum review candidates returned, ordered by priority; defaults to 10")
    ),
    responses(
        (status = 200, description = "Prioritized correction review queue for operator workflows", body = CorrectionReviewQueueResponse, example = json!({
            "total": 2,
            "entries": [{
                "target_kind": "belief",
                "target_id": "belief-1",
                "repeat_count": 2,
                "downstream_impact": 3,
                "priority_score": 6.1,
                "severity": "high"
            }]
        }))
    )
)]
pub(super) async fn correction_review_queue(
    State(state): State<ServerState>,
    Query(query): Query<CorrectionReviewQueueQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let entries = mem
        .get_correction_review_queue(query.limit)
        .into_iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(CorrectionReviewQueueResponse {
            total: entries.len(),
            entries,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/contradiction-review-queue",
    params(
        ("namespace" = Option<String>, Query, description = "Optional namespace filter"),
        ("limit" = Option<usize>, Query, description = "Maximum contradiction review candidates returned, ordered by priority")
    ),
    responses(
        (status = 200, description = "Prioritized contradiction review queue for unstable and unresolved belief clusters", body = ContradictionReviewQueueResponse, example = json!({
            "total": 1,
            "entries": [{
                "cluster_id": "cluster-1",
                "namespace": "default",
                "unresolved_belief_count": 2,
                "downstream_impact": 3,
                "priority_score": 5.7,
                "severity": "high"
            }]
        }))
    )
)]
pub(super) async fn contradiction_review_queue(
    State(state): State<ServerState>,
    Query(query): Query<ContradictionReviewQueueQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let entries = mem
        .get_contradiction_review_queue(query.namespace.as_deref(), query.limit)
        .into_iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(ContradictionReviewQueueResponse {
            total: entries.len(),
            entries,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/suggested-corrections",
    params(
        ("limit" = Option<usize>, Query, description = "Maximum suggested corrections returned, ordered by priority; defaults to 10")
    ),
    responses(
        (status = 200, description = "Advisory suggested corrections without auto-application", body = SuggestedCorrectionsResponse, example = json!({
            "scan_latency_ms": 3.4,
            "total": 2,
            "entries": [{
                "target_kind": "belief",
                "target_id": "belief-1",
                "reason_kind": "HighVolatility",
                "suggested_action": "Deprecate",
                "priority_score": 4.4
            }]
        }))
    )
)]
pub(super) async fn suggested_corrections(
    State(state): State<ServerState>,
    Query(query): Query<SuggestedCorrectionsQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let report = mem.get_suggested_corrections_report(query.limit);
    let entries = report
        .entries
        .into_iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(SuggestedCorrectionsResponse {
            scan_latency_ms: report.scan_latency_ms,
            total: entries.len(),
            entries,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/namespace-governance-status",
    params(
        ("namespaces" = Option<String>, Query, description = "Optional comma-separated namespaces to include")
    ),
    responses(
        (status = 200, description = "Read-only per-namespace governance summary", body = NamespaceGovernanceResponse, example = json!({
            "total": 2,
            "entries": [{
                "namespace": "alpha",
                "record_count": 12,
                "belief_count": 3,
                "correction_count": 2,
                "instability_level": "medium"
            }]
        }))
    )
)]
pub(super) async fn namespace_governance_status(
    State(state): State<ServerState>,
    Query(query): Query<NamespaceGovernanceQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let namespaces_owned = query.namespaces.as_ref().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|namespace| !namespace.is_empty())
            .collect::<Vec<_>>()
    });
    let entries = mem
        .get_namespace_governance_status_filtered(namespaces_owned.as_deref())
        .into_iter()
        .filter_map(|entry| serde_json::to_value(entry).ok())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(NamespaceGovernanceResponse {
            total: entries.len(),
            entries,
        }),
    )
}

#[utoipa::path(
    get,
    path = "/cross-namespace-digest",
    params(
        ("namespaces" = Option<String>, Query, description = "Comma-separated namespaces to include"),
        ("top_concepts_limit" = Option<usize>, Query, description = "Maximum concepts returned per namespace, clamped to 1..10"),
        ("min_record_count" = Option<usize>, Query, description = "Minimum record count required for a namespace to appear"),
        ("pairwise_similarity_threshold" = Option<f32>, Query, description = "Minimum pairwise similarity required for a pair entry to appear"),
        ("include_dimensions" = Option<String>, Query, description = "Comma-separated dimensions to include: concepts,tags,structural,causal,belief_states,corrections"),
        ("compact_summary" = Option<bool>, Query, description = "When true, omit bulky per-namespace and per-pair lists while keeping summaries and scores")
    ),
    responses((status = 200, description = "Cross-namespace analytics digest", body = serde_json::Value, example = json!({
        "namespace_count": 2,
        "compact_summary": false,
        "included_dimensions": ["concepts", "tags", "causal", "belief_states", "corrections"],
        "namespaces": [{"namespace": "alpha", "record_count": 12}],
        "pairs": [{"namespace_a": "alpha", "namespace_b": "beta", "tag_jaccard": 0.75}]
    })))
)]
pub(super) async fn cross_namespace_digest(
    State(state): State<ServerState>,
    Query(query): Query<CrossNamespaceDigestQuery>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let namespaces_owned = query.namespaces.as_ref().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|namespace| !namespace.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let namespaces_ref = namespaces_owned
        .as_ref()
        .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
    let include_dimensions_owned = query.include_dimensions.as_ref().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|dimension| !dimension.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let include_dimensions_ref = include_dimensions_owned
        .as_ref()
        .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
    let mut options = crate::aura::CrossNamespaceDigestOptions {
        min_record_count: query.min_record_count.unwrap_or(1),
        top_concepts_limit: query.top_concepts_limit.unwrap_or(5).clamp(1, 10),
        pairwise_similarity_threshold: query
            .pairwise_similarity_threshold
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
        compact_summary: query.compact_summary.unwrap_or(false),
        ..crate::aura::CrossNamespaceDigestOptions::default()
    };
    crate::aura::apply_cross_namespace_dimension_flags(
        &mut options,
        include_dimensions_ref.as_deref(),
    );
    let digest = mem.cross_namespace_digest_with_options(namespaces_ref.as_deref(), options);
    (StatusCode::OK, Json(serde_json::json!(digest)))
}

#[utoipa::path(post, path = "/batch-delete", request_body = BatchDeleteRequest, responses((status = 200, description = "Batch delete result", body = BatchDeleteResponse)))]
pub(super) async fn batch_delete(
    State(state): State<ServerState>,
    Json(payload): Json<BatchDeleteRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    let deleted = mem.batch_delete(&payload.ids);
    counter!("aura_delete_total", "status" => "ok").increment(deleted as u64);
    gauge!("aura_record_count").set(mem.count(None) as f64);
    (StatusCode::OK, Json(BatchDeleteResponse { deleted }))
}

#[utoipa::path(post, path = "/ingest-batch", request_body = IngestBatchRequest, responses((status = 200, description = "Batch ingestion result", body = IngestBatchResponse)))]
pub(super) async fn ingest_batch(
    State(state): State<ServerState>,
    Json(payload): Json<IngestBatchRequest>,
) -> impl IntoResponse {
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
            (
                StatusCode::OK,
                Json(IngestBatchResponse {
                    ingested: count,
                    pinned: payload.pinned,
                }),
            )
        }
        Err(_) => {
            counter!("aura_store_total", "status" => "error").increment(batch_size as u64);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(IngestBatchResponse {
                    ingested: 0,
                    pinned: payload.pinned,
                }),
            )
        }
    }
}

#[utoipa::path(post, path = "/predict", request_body = PredictRequest, responses((status = 200, description = "Temporal prediction", body = PredictResponse)))]
pub(super) async fn predict(
    State(state): State<ServerState>,
    Json(payload): Json<PredictRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.retrieve_prediction(&payload.id) {
        Ok(Some(rec)) => (
            StatusCode::OK,
            Json(PredictResponse {
                found: true,
                result: Some(StoredRecordDTO {
                    id: rec.id,
                    text: rec.text,
                    timestamp: rec.timestamp,
                    intensity: rec.intensity,
                    dna: rec.dna,
                    score: None,
                }),
            }),
        ),
        Ok(None) => (
            StatusCode::OK,
            Json(PredictResponse {
                found: false,
                result: None,
            }),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(PredictResponse {
                found: false,
                result: None,
            }),
        ),
    }
}

#[utoipa::path(post, path = "/surprise", request_body = SurpriseRequest, responses((status = 200, description = "Anomaly detection score", body = SurpriseResponse)))]
pub(super) async fn surprise_handler(
    State(state): State<ServerState>,
    Json(payload): Json<SurpriseRequest>,
) -> impl IntoResponse {
    let mem = state.memory.as_ref();
    match mem.surprise(&payload.predicted_id, &payload.actual_text) {
        Ok(val) => (StatusCode::OK, Json(SurpriseResponse { surprise: val })),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(SurpriseResponse { surprise: -1.0 }),
        ),
    }
}

#[cfg(feature = "sync")]
pub(super) async fn export_sdr(
    State(state): State<ServerState>,
    Json(payload): Json<ExportSdrRequest>,
) -> impl IntoResponse {
    use crate::sync::SdrPrivacyConfig;

    let mem = state.memory.as_ref();
    let config = SdrPrivacyConfig {
        apply_noise: payload.apply_noise,
        drop_bits: payload.drop_bits,
        add_bits: payload.add_bits,
    };
    let fps = mem.export_sdr_fingerprints(payload.filter_dna.as_deref(), &config);
    let count = fps.len();
    let dtos: Vec<SdrFingerprintDTO> = fps
        .into_iter()
        .map(|fp| SdrFingerprintDTO {
            id: fp.id,
            sdr_indices: fp.sdr_indices,
            timestamp: fp.timestamp,
            source_dna: fp.source_dna,
            intensity: fp.intensity,
            origin_node: fp.origin_node,
        })
        .collect();
    (
        StatusCode::OK,
        Json(ExportSdrResponse {
            fingerprints: dtos,
            count,
        }),
    )
}

#[cfg(feature = "sync")]
pub(super) async fn import_sdr(
    State(state): State<ServerState>,
    Json(payload): Json<ImportSdrRequest>,
) -> impl IntoResponse {
    use crate::sync::SdrFingerprint;

    let mem = state.memory.as_ref();
    let fps: Vec<SdrFingerprint> = payload
        .fingerprints
        .into_iter()
        .map(|f| SdrFingerprint {
            id: f.id,
            sdr_indices: f.sdr_indices,
            timestamp: f.timestamp,
            source_dna: f.source_dna,
            intensity: f.intensity,
            origin_node: f.origin_node,
        })
        .collect();
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
        health,
        process,
        retrieve,
        memory_health,
        belief_instability,
        policy_lifecycle,
        explain_record,
        explain_recall,
        explainability_bundle,
        correction_log,
        correction_review_queue,
        contradiction_review_queue,
        suggested_corrections,
        namespace_governance_status,
        delete_memory,
        update_memory,
        stats,
        list_memories,
        analytics,
        cross_namespace_digest,
        batch_delete,
        ingest_batch,
        predict,
        surprise_handler
    ),
    components(schemas(
        ProcessRequest,
        ProcessResponse,
        RetrieveRequest,
        RetrieveResponse,
        StoredRecordDTO,
        DeleteRequest,
        DeleteResponse,
        UpdateRequest,
        StatsResponse,
        MemoriesQuery,
        MemoriesResponse,
        AnalyticsResponse,
        ExplainRecordQuery,
        ExplainRecallQuery,
        CorrectionLogQuery,
        CorrectionReviewQueueQuery,
        ContradictionReviewQueueQuery,
        SuggestedCorrectionsQuery,
        NamespaceGovernanceQuery,
        ExplainRecordResponse,
        ExplainRecallResponse,
        ExplainabilityBundleResponse,
        CorrectionLogResponse,
        CorrectionReviewQueueResponse,
        ContradictionReviewQueueResponse,
        SuggestedCorrectionsResponse,
        NamespaceGovernanceResponse,
        CrossNamespaceDigestQuery,
        MemoryHealthQuery,
        BeliefInstabilityQuery,
        PolicyLifecycleQuery,
        BatchDeleteRequest,
        BatchDeleteResponse,
        IngestBatchRequest,
        IngestBatchResponse,
        MemoryHealthResponse,
        BeliefInstabilityResponse,
        PolicyLifecycleResponse,
        PredictRequest,
        PredictResponse,
        SurpriseRequest,
        SurpriseResponse,
    ))
)]
pub(super) struct ApiDoc;
