use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use metrics::{counter, histogram};

use super::state::{ServerState, API_PATHS};

pub(super) async fn auth_middleware(
    State(state): State<ServerState>,
    req: Request,
    next: Next,
) -> Response {
    let expected = match state.api_key.as_deref() {
        Some(k) => k,
        None => return next.run(req).await,
    };

    let path = req.uri().path().to_string();
    let is_api = API_PATHS.iter().any(|p| path.starts_with(p));
    if !is_api {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
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

pub(super) async fn rate_limit_middleware(
    State(state): State<ServerState>,
    req: Request,
    next: Next,
) -> Response {
    let max = state.rate_limit.max.load(Ordering::Relaxed);
    if max == 0 {
        return next.run(req).await;
    }

    let should_allow = {
        let mut window = state.rate_limit.window_start.lock().unwrap();
        let now = Instant::now();
        if now.duration_since(*window).as_secs() >= 1 {
            *window = now;
            state.rate_limit.counter.store(1, Ordering::Relaxed);
            true
        } else {
            let count = state.rate_limit.counter.fetch_add(1, Ordering::Relaxed);
            count < max
        }
    };

    if should_allow {
        next.run(req).await
    } else {
        StatusCode::TOO_MANY_REQUESTS.into_response()
    }
}

pub(super) async fn metrics_middleware(req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let method = req.method().to_string();
    let start = Instant::now();

    let response = next.run(req).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    counter!("aura_http_requests_total", "method" => method.clone(), "path" => path.clone(), "status" => status).increment(1);
    histogram!("aura_http_request_duration_seconds", "method" => method, "path" => path)
        .record(duration);

    response
}
