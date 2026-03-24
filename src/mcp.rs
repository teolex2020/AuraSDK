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

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct CrossNamespaceDigestParams {
    /// Optional subset of namespaces to include. Omit for all namespaces.
    namespaces: Option<Vec<String>>,
    /// Maximum concepts returned per namespace. Clamped to 1..10.
    top_concepts_limit: Option<usize>,
    /// Minimum record count required for a namespace to appear.
    min_record_count: Option<usize>,
    /// Minimum pairwise similarity required for a pair entry to appear.
    pairwise_similarity_threshold: Option<f32>,
    /// Which dimensions to include. Supported: concepts,tags,structural,causal,belief_states,corrections.
    include_dimensions: Option<Vec<String>>,
    /// When true, omit bulky detail lists while keeping summaries and scores.
    compact_summary: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExplainRecordParams {
    /// Record ID to explain.
    record_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExplainRecallParams {
    /// Natural-language query to explain.
    query: String,
    /// Maximum number of results to explain.
    top_k: Option<usize>,
    /// Minimum record strength required for inclusion.
    min_strength: Option<f32>,
    /// Whether graph/context expansion is enabled.
    expand_connections: Option<bool>,
    /// Namespace to search in (default: "default").
    namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorrectionLogParams {
    /// Optional target kind filter: belief, causal_pattern, policy_hint, record.
    target_kind: Option<String>,
    /// Optional target ID filter; only applied together with target_kind.
    target_id: Option<String>,
    /// Maximum entries returned, newest first.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorrectionReviewQueueParams {
    /// Maximum review candidates returned, ordered by priority.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ContradictionReviewQueueParams {
    /// Optional namespace filter.
    namespace: Option<String>,
    /// Maximum review candidates returned, ordered by priority.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SuggestedCorrectionsParams {
    /// Maximum suggested corrections returned, ordered by priority.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NamespaceGovernanceParams {
    /// Optional subset of namespaces to include. Omit for all namespaces.
    namespaces: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PolicyLifecycleParams {
    /// Optional namespace filter for list-style policy tools.
    namespace: Option<String>,
    /// Maximum number of returned items.
    limit: Option<usize>,
    /// Maximum action summary rows.
    action_limit: Option<usize>,
    /// Maximum domain summary rows.
    domain_limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BeliefInstabilityParams {
    /// Minimum volatility threshold for high-volatility beliefs.
    min_volatility: Option<f32>,
    /// Maximum stability threshold for low-stability beliefs.
    max_stability: Option<f32>,
    /// Maximum number of returned items.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MemoryHealthParams {
    /// Maximum top issues returned.
    limit: Option<usize>,
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
        description = "Return a read-only digest across namespaces: top concepts, shared tags, structural overlap, and canonical causal-signature overlap."
    )]
    async fn cross_namespace_digest(
        &self,
        Parameters(p): Parameters<CrossNamespaceDigestParams>,
    ) -> Result<CallToolResult, McpError> {
        let namespaces_ref = p
            .namespaces
            .as_ref()
            .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
        let include_dimensions_ref = p
            .include_dimensions
            .as_ref()
            .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
        let mut options = crate::aura::CrossNamespaceDigestOptions {
            min_record_count: p.min_record_count.unwrap_or(1),
            top_concepts_limit: p.top_concepts_limit.unwrap_or(5).clamp(1, 10),
            pairwise_similarity_threshold: p
                .pairwise_similarity_threshold
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            compact_summary: p.compact_summary.unwrap_or(false),
            ..crate::aura::CrossNamespaceDigestOptions::default()
        };
        crate::aura::apply_cross_namespace_dimension_flags(
            &mut options,
            include_dimensions_ref.as_deref(),
        );
        let digest = self
            .brain
            .cross_namespace_digest_with_options(namespaces_ref.as_deref(), options);
        let json = serde_json::to_string(&digest).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Explain one record through belief, concept, causal, and policy provenance. Use for operator debugging of a specific memory item."
    )]
    async fn explain_record(
        &self,
        Parameters(p): Parameters<ExplainRecordParams>,
    ) -> Result<CallToolResult, McpError> {
        let item = self
            .brain
            .explain_record(&p.record_id)
            .ok_or_else(|| err(format!("record not found: {}", p.record_id)))?;
        let json = serde_json::to_string(&item).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Explain recall scoring and provenance for a query. Returns bounded trace-scoring output for the retrieved results."
    )]
    async fn explain_recall(
        &self,
        Parameters(p): Parameters<ExplainRecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let ns_vec: Option<Vec<&str>> = p.namespace.as_ref().map(|s| vec![s.as_str()]);
        let ns_slice: Option<&[&str]> = ns_vec.as_deref();
        let explanation = self.brain.explain_recall(
            &p.query,
            p.top_k,
            p.min_strength,
            p.expand_connections,
            ns_slice,
        );
        let json = serde_json::to_string(&explanation).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return one bounded explainability bundle for a record: direct explanation, provenance chain, correction excerpts, instability, and maintenance-trend summary."
    )]
    async fn explainability_bundle(
        &self,
        Parameters(p): Parameters<ExplainRecordParams>,
    ) -> Result<CallToolResult, McpError> {
        let bundle = self
            .brain
            .explainability_bundle(&p.record_id)
            .ok_or_else(|| err(format!("record not found: {}", p.record_id)))?;
        let json = serde_json::to_string(&bundle).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Read correction-log entries globally or for a specific target. Use to audit manual deprecations, invalidations, and retractions."
    )]
    async fn correction_log(
        &self,
        Parameters(p): Parameters<CorrectionLogParams>,
    ) -> Result<CallToolResult, McpError> {
        let max = p.limit.unwrap_or(50).min(200);
        let mut entries = if let (Some(target_kind), Some(target_id)) =
            (p.target_kind.as_deref(), p.target_id.as_deref())
        {
            self.brain
                .get_correction_log_for_target(target_kind, target_id)
        } else {
            self.brain.get_correction_log()
        };
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(max);
        let json = serde_json::to_string(&entries).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return a prioritized correction review queue using recency, repeated corrections, and downstream causal/policy impact."
    )]
    async fn correction_review_queue(
        &self,
        Parameters(p): Parameters<CorrectionReviewQueueParams>,
    ) -> Result<CallToolResult, McpError> {
        let queue = self.brain.get_correction_review_queue(p.limit);
        let json = serde_json::to_string(&queue).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return a prioritized contradiction review queue using unstable belief clusters, conflict mass, and downstream causal/policy impact."
    )]
    async fn contradiction_review_queue(
        &self,
        Parameters(p): Parameters<ContradictionReviewQueueParams>,
    ) -> Result<CallToolResult, McpError> {
        let queue = self
            .brain
            .get_contradiction_review_queue(p.namespace.as_deref(), p.limit);
        let json = serde_json::to_string(&queue).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return advisory suggested corrections without auto-application. Uses instability, rejected causal patterns, suppressed policy hints, and review pressure."
    )]
    async fn suggested_corrections(
        &self,
        Parameters(p): Parameters<SuggestedCorrectionsParams>,
    ) -> Result<CallToolResult, McpError> {
        let report = self.brain.get_suggested_corrections_report(p.limit);
        let json = serde_json::to_string(&report).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return read-only per-namespace governance summaries: record count, belief count, correction pressure, instability level, and latest maintenance cycle."
    )]
    async fn namespace_governance_status(
        &self,
        Parameters(p): Parameters<NamespaceGovernanceParams>,
    ) -> Result<CallToolResult, McpError> {
        let namespaces_ref = p
            .namespaces
            .as_ref()
            .map(|items| items.iter().map(String::as_str).collect::<Vec<_>>());
        let statuses = self
            .brain
            .get_namespace_governance_status_filtered(namespaces_ref.as_deref());
        let json = serde_json::to_string(&statuses).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return bounded policy lifecycle summaries and advisory-pressure areas for operator inspection."
    )]
    async fn policy_lifecycle(
        &self,
        Parameters(p): Parameters<PolicyLifecycleParams>,
    ) -> Result<CallToolResult, McpError> {
        let summary = self
            .brain
            .get_policy_lifecycle_summary(p.action_limit, p.domain_limit);
        let pressure = self
            .brain
            .get_policy_pressure_report(p.namespace.as_deref(), p.limit);
        let suppressed = self
            .brain
            .get_suppressed_policy_hints(p.namespace.as_deref(), p.limit);
        let rejected = self
            .brain
            .get_rejected_policy_hints(p.namespace.as_deref(), p.limit);
        let payload = serde_json::json!({
            "summary": summary,
            "pressure": pressure,
            "suppressed": suppressed,
            "rejected": rejected,
        });
        let json = serde_json::to_string(&payload).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return bounded belief-instability inspection output: summary, high-volatility beliefs, low-stability beliefs, and recently corrected beliefs."
    )]
    async fn belief_instability(
        &self,
        Parameters(p): Parameters<BeliefInstabilityParams>,
    ) -> Result<CallToolResult, McpError> {
        let payload = serde_json::json!({
            "summary": self.brain.get_belief_instability_summary(),
            "high_volatility": self.brain.get_high_volatility_beliefs(p.min_volatility, p.limit),
            "low_stability": self.brain.get_low_stability_beliefs(p.max_stability, p.limit),
            "recently_corrected": self.brain.get_recently_corrected_beliefs(p.limit),
        });
        let json = serde_json::to_string(&payload).map_err(|e| err(e.to_string()))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Return one compact operator-facing memory health digest: corrections, instability, policy pressure, startup recovery warnings, and maintenance trend direction."
    )]
    async fn memory_health(
        &self,
        Parameters(p): Parameters<MemoryHealthParams>,
    ) -> Result<CallToolResult, McpError> {
        let digest = self.brain.get_memory_health_digest(p.limit);
        let json = serde_json::to_string(&digest).map_err(|e| err(e.to_string()))?;
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
