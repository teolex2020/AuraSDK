//! MCP (Model Context Protocol) server for Aura.
//!
//! Exposes Aura memory operations as MCP tools for Claude Desktop,
//! Claude Code, Gemini, and any MCP-compatible client.
//!
//! Feature-gated behind `mcp`.
//!
//! # Environment variables
//! - `AURA_BRAIN_PATH` — path to brain storage (default: `./aura_brain`)
//! - `AURA_PASSWORD`   — optional encryption password

use std::env;
use std::sync::Arc;

use rmcp::schemars::JsonSchema;
use rmcp::{
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{
        CallToolResult, Content, Implementation, InitializeResult, ProtocolVersion,
        ServerCapabilities, ServerInfo,
    },
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::Deserialize;

use crate::aura::Aura;
use crate::levels::Level;

// ── Tool parameter schemas ──

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Natural language query to search memories.
    query: String,
    /// Maximum tokens in output (default: 2048).
    token_budget: Option<usize>,
    /// Namespace to search in (default: "default").
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallStructuredParams {
    /// Natural language query to search memories.
    query: String,
    /// Maximum number of results (default: 20).
    top_k: Option<usize>,
    /// Namespace to search in (default: "default").
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreParams {
    /// The text content to store.
    content: String,
    /// Memory level: working, decisions, domain, or identity.
    level: Option<String>,
    /// Tags for categorization.
    tags: Option<Vec<String>>,
    /// Content type hint (text, code, decision).
    content_type: Option<String>,
    /// How the data was obtained: "recorded", "retrieved", "inferred", "generated".
    source_type: Option<String>,
    /// ID of the record that caused this one.
    caused_by_id: Option<String>,
    /// Namespace to store in (default: "default").
    namespace: Option<String>,
    /// Semantic role: "fact", "decision", "trend", "serendipity", "preference", "contradiction".
    semantic_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreCodeParams {
    /// The source code to store.
    code: String,
    /// Programming language (e.g., python, rust, javascript).
    language: String,
    /// Optional filename.
    filename: Option<String>,
    /// Tags for categorization.
    tags: Option<Vec<String>>,
    /// Namespace to store in (default: "default").
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreDecisionParams {
    /// The decision that was made.
    decision: String,
    /// Reasoning behind the decision.
    reasoning: Option<String>,
    /// Alternatives that were considered.
    alternatives: Option<Vec<String>>,
    /// Tags for categorization.
    tags: Option<Vec<String>>,
    /// ID of the record that caused this decision.
    caused_by_id: Option<String>,
    /// Namespace to store in (default: "default").
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Text substring to match.
    query: Option<String>,
    /// Filter by level: working, decisions, domain, identity.
    level: Option<String>,
    /// Filter by tags (record must have all specified tags).
    tags: Option<Vec<String>>,
    /// Filter by content type.
    content_type: Option<String>,
    /// Filter by source type: "recorded", "retrieved", "inferred", "generated".
    source_type: Option<String>,
    /// Namespace to search in (default: "default").
    namespace: Option<String>,
    /// Filter by semantic type: "fact", "decision", "trend", "serendipity", "preference", "contradiction".
    semantic_type: Option<String>,
}

// ── Helper ──

fn parse_level(s: &str) -> Option<Level> {
    match s.to_lowercase().as_str() {
        "working" => Some(Level::Working),
        "decisions" => Some(Level::Decisions),
        "domain" => Some(Level::Domain),
        "identity" => Some(Level::Identity),
        _ => None,
    }
}

fn err(msg: impl Into<String>) -> McpError {
    McpError::internal_error(msg.into(), None)
}

// ── MCP Server ──

#[derive(Clone)]
pub struct AuraMcpServer {
    brain: Arc<Aura>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl AuraMcpServer {
    pub fn new(brain: Arc<Aura>) -> Self {
        Self {
            brain,
            tool_router: Self::tool_router(),
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let path = env::var("AURA_BRAIN_PATH").unwrap_or_else(|_| "./aura_brain".into());
        let password = env::var("AURA_PASSWORD").ok();
        let brain = if let Some(pw) = password {
            Aura::open_with_password(&path, Some(&pw))?
        } else {
            Aura::open(&path)?
        };
        Ok(Self::new(Arc::new(brain)))
    }

    // ── Tools ──

    #[tool(
        description = "Retrieve relevant memories as context for a query. Call BEFORE answering to check existing knowledge. Returns formatted context for LLM injection."
    )]
    async fn recall(
        &self,
        Parameters(p): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let ns_vec: Option<Vec<&str>> = p.namespace.as_ref().map(|s| vec![s.as_str()]);
        let ns_slice: Option<&[&str]> = ns_vec.as_deref();
        let result = self
            .brain
            .recall(&p.query, p.token_budget, None, None, None, ns_slice)
            .map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        description = "Retrieve memories as structured data with scores. Use when you need individual records with scores, levels, and metadata."
    )]
    async fn recall_structured(
        &self,
        Parameters(p): Parameters<RecallStructuredParams>,
    ) -> Result<CallToolResult, McpError> {
        let ns_vec: Option<Vec<&str>> = p.namespace.as_ref().map(|s| vec![s.as_str()]);
        let ns_slice: Option<&[&str]> = ns_vec.as_deref();
        let results = self
            .brain
            .recall_structured(&p.query, p.top_k, None, None, None, ns_slice)
            .map_err(|e| err(e.to_string()))?;
        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|(score, rec)| {
                serde_json::json!({
                    "id": rec.id,
                    "content": rec.content,
                    "score": score,
                    "level": format!("{:?}", rec.level),
                    "tags": rec.tags,
                    "strength": rec.strength,
                    "source_type": rec.source_type,
                    "semantic_type": rec.semantic_type,
                })
            })
            .collect();
        let json = serde_json::to_string(&items).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Store a new memory. Levels: working (hours), decisions (days), domain (weeks), identity (months+). Auto-detects novel info and boosts level."
    )]
    async fn store(
        &self,
        Parameters(p): Parameters<StoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let level = p.level.as_deref().and_then(parse_level);
        let rec = self
            .brain
            .store(
                &p.content,
                level,
                p.tags,
                None,
                p.content_type.as_deref(),
                p.source_type.as_deref(),
                None,
                None,
                p.caused_by_id.as_deref(),
                p.namespace.as_deref(),
                p.semantic_type.as_deref(),
            )
            .map_err(|e| err(e.to_string()))?;
        let resp = serde_json::json!({"id": rec.id, "level": format!("{:?}", rec.level)});
        Ok(CallToolResult::success(vec![Content::text(
            resp.to_string(),
        )]))
    }

    #[tool(
        description = "Store a code snippet at DOMAIN level with language metadata and syntax highlighting in recall."
    )]
    async fn store_code(
        &self,
        Parameters(p): Parameters<StoreCodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut tags = p.tags.unwrap_or_default();
        tags.push("code".into());
        tags.push(p.language.clone());
        if let Some(ref f) = p.filename {
            tags.push(format!("file:{}", f));
        }
        let content = format!("```{}\n{}\n```", p.language, p.code);
        let rec = self
            .brain
            .store(
                &content,
                Some(Level::Domain),
                Some(tags),
                None,
                Some("code"),
                None,
                None,
                None,
                None,
                p.namespace.as_deref(),
                None,
            )
            .map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::json!({"id": rec.id, "level": "DOMAIN"}).to_string(),
        )]))
    }

    #[tool(
        description = "Store a decision with reasoning and rejected alternatives at DECISIONS level."
    )]
    async fn store_decision(
        &self,
        Parameters(p): Parameters<StoreDecisionParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut content = format!("DECISION: {}", p.decision);
        if let Some(ref r) = p.reasoning {
            if !r.is_empty() {
                content.push_str(&format!("\nREASONING: {}", r));
            }
        }
        if let Some(ref a) = p.alternatives {
            if !a.is_empty() {
                content.push_str(&format!("\nALTERNATIVES: {}", a.join(", ")));
            }
        }
        let mut tags = p.tags.unwrap_or_default();
        tags.push("decision".into());
        let rec = self
            .brain
            .store(
                &content,
                Some(Level::Decisions),
                Some(tags),
                None,
                None,
                None,
                None,
                None,
                p.caused_by_id.as_deref(),
                p.namespace.as_deref(),
                Some("decision"),
            )
            .map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::json!({"id": rec.id, "level": "DECISIONS"}).to_string(),
        )]))
    }

    #[tool(
        description = "Search memory by filters (exact/tag-based, not ranked). Use for browsing or counting."
    )]
    async fn search(
        &self,
        Parameters(p): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let level = p.level.as_deref().and_then(parse_level);
        let ns_vec: Option<Vec<&str>> = p.namespace.as_ref().map(|s| vec![s.as_str()]);
        let ns_slice: Option<&[&str]> = ns_vec.as_deref();
        let results = self.brain.search(
            p.query.as_deref(),
            level,
            p.tags,
            None,
            p.content_type.as_deref(),
            p.source_type.as_deref(),
            ns_slice,
            p.semantic_type.as_deref(),
        );
        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id, "content": r.content,
                    "level": format!("{:?}", r.level), "tags": r.tags,
                    "semantic_type": r.semantic_type,
                })
            })
            .collect();
        let json = serde_json::to_string(&items).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Get proactive insights about memory health. Detects decay risks, promotion candidates, clusters, conflicts, and trends."
    )]
    async fn insights(&self) -> Result<CallToolResult, McpError> {
        let stats = self.brain.stats();
        let json = serde_json::to_string(&stats).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Merge similar memory records (85%+ similarity) to reduce bloat. Call periodically for hygiene."
    )]
    async fn consolidate(&self) -> Result<CallToolResult, McpError> {
        let result = self.brain.consolidate().map_err(|e| err(e.to_string()))?;
        let resp = serde_json::json!({
            "merged": result.get("merged").copied().unwrap_or(0),
            "checked": result.get("checked").copied().unwrap_or(0),
        });
        Ok(CallToolResult::success(vec![Content::text(
            resp.to_string(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for AuraMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_06_18,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "aura".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                website_url: None,
                icons: None,
            },
            instructions: Some(
                "Aura is a cognitive memory layer for AI agents. \
                 It provides hierarchical memory with 4 levels: \
                 working (hours), decisions (days), domain (weeks), identity (months+). \
                 Use 'recall' before answering to check existing context. \
                 Use 'store' to remember facts, decisions, and patterns. \
                 Use 'store_code' for code snippets. \
                 Use 'store_decision' for decisions with reasoning. \
                 Use 'insights' to check memory health."
                    .into(),
            ),
        }
    }

    async fn initialize(
        &self,
        _request: rmcp::model::InitializeRequestParam,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

/// Run the MCP server with stdio transport.
pub async fn run_stdio() -> anyhow::Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    tracing::info!("Starting Aura MCP server (stdio)");
    let server = AuraMcpServer::from_env()?;
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
