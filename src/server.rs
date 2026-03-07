mod dto;
mod handlers;
mod middleware;
mod state;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use axum::{
    http::header,
    middleware as axum_middleware,
    routing::{get, post},
    Router,
};
use tower_http::cors::{AllowOrigin, CorsLayer};
use utoipa::OpenApi;

use crate::aura::Aura;

use self::handlers::ApiDoc;
use self::state::{RateLimitState, ServerState};

pub fn start_server(port: u16, storage_path: &str) -> anyhow::Result<()> {
    let api_key = std::env::var("AURA_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .map(Arc::<str>::from);
    if api_key.is_some() {
        println!("[AUTH] API Key authentication enabled");
    }

    let rate_limit_max = std::env::var("AURA_RATE_LIMIT")
        .ok()
        .and_then(|val| val.parse::<u64>().ok())
        .unwrap_or(100);
    if rate_limit_max > 0 {
        println!("[RATE] Rate limit: {} req/s", rate_limit_max);
    } else {
        println!("[RATE] Rate limiting disabled");
    }

    let bind_addr: SocketAddr = std::env::var("AURA_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], port)));

    let cors_layer = match std::env::var("AURA_CORS_ORIGINS").ok() {
        Some(val) if val == "*" => {
            println!("[CORS] Allowing all origins (development mode)");
            CorsLayer::permissive()
        }
        Some(val) if !val.is_empty() => {
            let origins: Vec<axum::http::HeaderValue> = val.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            println!("[CORS] Allowed origins: {} configured", origins.len());
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

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder();
    let prom_handle = recorder.handle();
    metrics::set_global_recorder(recorder).ok();
    println!("[METRICS] Prometheus metrics enabled at /metrics");

    let state = ServerState {
        memory: Arc::new(Aura::open(storage_path)?),
        api_key,
        rate_limit: Arc::new(RateLimitState {
            counter: AtomicU64::new(0),
            window_start: Mutex::new(Instant::now()),
            max: AtomicU64::new(rate_limit_max),
        }),
        prom_handle: Arc::new(prom_handle),
    };

    metrics::gauge!("aura_record_count").set(state.memory.count(None) as f64);

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
        let api = Router::new()
            .route("/health", get(handlers::health))
            .route("/process", post(handlers::process))
            .route("/retrieve", post(handlers::retrieve))
            .route("/delete", post(handlers::delete_memory))
            .route("/update", post(handlers::update_memory))
            .route("/stats", get(handlers::stats))
            .route("/memories", get(handlers::list_memories))
            .route("/analytics", get(handlers::analytics))
            .route("/batch-delete", post(handlers::batch_delete))
            .route("/ingest-batch", post(handlers::ingest_batch))
            .route("/predict", post(handlers::predict))
            .route("/surprise", post(handlers::surprise_handler));

        #[cfg(feature = "sync")]
        let api = api
            .route("/export-sdr", post(handlers::export_sdr))
            .route("/import-sdr", post(handlers::import_sdr));

        let api = api
            .layer(axum_middleware::from_fn_with_state(state.clone(), middleware::rate_limit_middleware))
            .layer(axum_middleware::from_fn_with_state(state.clone(), middleware::auth_middleware));

        let app = api
            .merge(
                utoipa_swagger_ui::SwaggerUi::new("/docs")
                    .url("/openapi.json", ApiDoc::openapi())
            )
            .route("/metrics", get(handlers::prometheus_metrics))
            .route("/", get(handlers::static_handler))
            .route("/*file", get(handlers::static_handler))
            .layer(axum_middleware::from_fn(middleware::metrics_middleware))
            .layer(cors_layer)
            .with_state(state.clone());

        let tls_cert = std::env::var("AURA_TLS_CERT").ok();
        let tls_key = std::env::var("AURA_TLS_KEY").ok();
        let use_tls = tls_cert.is_some() && tls_key.is_some();

        let scheme = if use_tls { "https" } else { "http" };
        println!("Server listening at {}://{}", scheme, bind_addr);

        if bind_addr.ip().is_loopback() && !use_tls {
            if let Err(_e) = open::that(format!("http://{}", bind_addr)) {}
        }

        if use_tls {
            let cert_path = tls_cert.unwrap();
            let key_path = tls_key.unwrap();
            println!("[TLS] Loading cert: {}, key: {}", cert_path, key_path);

            let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                .await
                .expect("Failed to load TLS cert/key. Check AURA_TLS_CERT and AURA_TLS_KEY paths.");

            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();

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
            let listener = tokio::net::TcpListener::bind(bind_addr).await.unwrap();
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .unwrap();
        }

        println!("\n[SHUTDOWN] Flushing memory to disk...");
        let _ = state.memory.close();
        println!("[SHUTDOWN] All data saved. Goodbye.");
    });

    Ok(())
}

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

mod open {
    pub fn that<T: AsRef<std::ffi::OsStr>>(path: T) -> std::io::Result<()> {
        std::process::Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path.as_ref())
            .spawn()?;
        Ok(())
    }
}
