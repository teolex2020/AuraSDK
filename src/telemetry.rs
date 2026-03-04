//! OpenTelemetry integration for distributed tracing.
//!
//! Exports spans to any OTLP-compatible collector (Jaeger, Tempo, Grafana Cloud).
//!
//! # Environment Variables
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (default: `http://localhost:4317`)
//! - `OTEL_SERVICE_NAME` — Service name (default: `aura`)
//! - `OTEL_RESOURCE_ATTRIBUTES` — Additional resource attributes (key=value,key=value)
//!
//! # Usage
//! ```rust,ignore
//! let provider = aura::telemetry::init_telemetry()?;
//! // ... all #[instrument] spans will be exported
//! aura::telemetry::shutdown_telemetry(provider);
//! ```

use anyhow::Result;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::TracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initialize the OpenTelemetry tracing pipeline.
///
/// Sets up an OTLP exporter that sends spans to the configured endpoint.
/// Must be called before any traced operations. Safe to call multiple times
/// (subsequent calls are no-ops if a global subscriber is already set).
pub fn init_telemetry() -> Result<TracerProvider> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .build();

    let tracer = provider.tracer("aura");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    // Try to set global subscriber — if one already exists, this is a no-op
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(otel_layer)
        .with(tracing_subscriber::fmt::layer())
        .try_init();

    Ok(provider)
}

/// Gracefully shut down the telemetry pipeline, flushing pending spans.
pub fn shutdown_telemetry(provider: TracerProvider) {
    let _ = provider.shutdown();
}
