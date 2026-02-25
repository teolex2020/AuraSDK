//! Aura MCP Server binary.
//!
//! Runs the Aura memory layer as an MCP server over stdio transport.
//! Compatible with Claude Desktop, Claude Code, and any MCP client.
//!
//! # Environment variables
//! - `AURA_BRAIN_PATH` — path to brain storage (default: `./aura_brain`)
//! - `AURA_PASSWORD`   — optional encryption password
//!
//! # Usage
//!     aura-mcp

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .init();

    aura::mcp::run_stdio().await
}
