# Aura API Reference

## Classes

### `Aura(path, password=None)`

Main entry point. Creates or opens a cognitive memory brain.

**Parameters:**
- `path` (str) — Path to the data directory. Created if it doesn't exist.
- `password` (str, optional) — Enable encryption with ChaCha20-Poly1305. Key derived via Argon2id.

```python
from aura import Aura
brain = Aura("./data")                          # unencrypted
brain = Aura("./secret", password="my-pass")    # encrypted
```

---

## Core Methods

### `store(content, level=None, tags=None, pin=None, content_type=None, metadata=None, deduplicate=None, caused_by_id=None, channel=None) -> str`

Store a memory record. Returns the record ID.

**Parameters:**
- `content` (str) — The text content to store (max 100KB).
- `level` (Level, optional) — Memory level. Default: auto-detected (Working if novel, promoted if familiar).
- `tags` (list[str], optional) — Tags for categorization. Max 50 tags.
- `pin` (bool, optional) — Pin record to prevent decay.
- `content_type` (str, optional) — Content type hint (e.g., "code", "decision").
- `metadata` (dict[str, str], optional) — Arbitrary key-value metadata.
- `deduplicate` (bool, optional) — Enable deduplication check. Default: True.
- `caused_by_id` (str, optional) — Link to the record that caused this one.
- `channel` (str, optional) — Source channel for provenance stamping (e.g., "user", "api", "telegram").

```python
record_id = brain.store(
    "User prefers dark mode",
    level=Level.Identity,
    tags=["preference", "ui"],
    channel="user"
)
```

**Automatic behaviors:**
- Guards auto-detect phone numbers, emails, wallets, API keys and add protective tags.
- Provenance is stamped with source, trust score, volatility, and timestamp.
- Deduplication merges similar content (SDR similarity > 80%).
- Novel information (surprise > threshold) may be auto-promoted.

---

### `recall(query, token_budget=None, min_strength=None, expand_connections=None, session_id=None) -> str`

Recall memories as formatted context for LLM injection. Returns a formatted string grouped by level.

**Parameters:**
- `query` (str) — Natural language query.
- `token_budget` (int, optional) — Maximum tokens in output. Default: 4096.
- `min_strength` (float, optional) — Minimum strength threshold (0.0–1.0).
- `expand_connections` (bool, optional) — Include connected records. Default: True.
- `session_id` (str, optional) — Session ID for session-aware recall.

```python
context = brain.recall("authentication issues", token_budget=2000)
# Returns:
# === COGNITIVE CONTEXT ===
# [DOMAIN]
#   - OAuth2 flow uses PKCE for mobile [auth, oauth]
# [WORKING]
#   - Bug in auth module needs fix [bug, auth]
```

---

### `recall_structured(query, top_k=None, min_strength=None, expand_connections=None, session_id=None) -> list[dict]`

Recall memories as structured data with scores. Returns a list of dicts.

**Parameters:**
- `query` (str) — Natural language query.
- `top_k` (int, optional) — Maximum results. Default: 20.
- `min_strength` (float, optional) — Minimum strength threshold.
- `expand_connections` (bool, optional) — Include connected records. Default: True.
- `session_id` (str, optional) — Session ID for session-aware recall.

**Returns:** List of dicts with keys:
- `id` (str) — Record ID.
- `content` (str) — Record content.
- `score` (float) — RRF fusion score, weighted by trust.
- `level` (str) — Memory level ("WORKING", "DECISIONS", "DOMAIN", "IDENTITY").
- `strength` (float) — Current strength (0.0–1.0).
- `tags` (list[str]) — Record tags.
- `trust` (str) — Effective trust score.
- `source` (str) — Provenance source.

```python
results = brain.recall_structured("rust performance", top_k=5)
for r in results:
    print(f"[{r['level']}] score={r['score']:.3f} — {r['content'][:60]}")
```

---

### `search(query=None, level=None, tags=None, content_type=None) -> list[Record]`

Filter-based search (not ranked). Returns matching Record objects.

**Parameters:**
- `query` (str, optional) — Text substring to match.
- `level` (Level, optional) — Filter by level.
- `tags` (list[str], optional) — Filter by tags (records must have all specified tags).
- `content_type` (str, optional) — Filter by content type.

```python
records = brain.search(tags=["bug", "auth"])
records = brain.search(level=Level.Identity)
records = brain.search(query="dark mode")
```

---

### `get(record_id) -> Record | None`

Get a record by ID. Returns None if not found.

```python
rec = brain.get("abc123")
if rec:
    print(rec.content, rec.level, rec.tags)
```

---

### `update(record_id, content=None, level=None, tags=None, strength=None, metadata=None) -> Record | None`

Update a record's fields. Only provided fields are changed. Returns the updated Record or None if not found.

```python
brain.update(record_id, content="Updated content", tags=["new", "tags"])
brain.update(record_id, level=Level.Domain)
brain.update(record_id, strength=1.0)  # Re-strengthen a decaying record
```

---

### `delete(record_id) -> bool`

Delete a record. Returns True if deleted, False if not found.

```python
brain.delete("abc123")
```

---

### `connect(id_a, id_b, weight=None, relationship=None)`

Create a typed connection between two records in the knowledge graph.

**Parameters:**
- `id_a` (str) — First record ID.
- `id_b` (str) — Second record ID.
- `weight` (float, optional) — Connection strength (0.0–1.0). Default: 0.5.
- `relationship` (str, optional) — Connection type. Built-in types: `"causal"`, `"reflective"`, `"associative"`, `"coactivation"`. Custom types are also supported.

```python
brain.connect(id1, id2, weight=0.8)                          # untyped
brain.connect(id1, id2, relationship="causal")                # typed
brain.connect(id1, id2, weight=0.9, relationship="reflective")  # typed with weight
```

**Automatic connections:** Aura creates typed connections automatically:
- `"associative"` — records sharing tags are auto-connected on store
- `"coactivation"` — records recalled in the same session get strengthened
- `"causal"` — records linked via `caused_by_id` on store

---

### `stats() -> dict`

Get brain statistics.

```python
stats = brain.stats()
# {'total_records': 42, 'working': 20, 'decisions': 10,
#  'domain': 8, 'identity': 4, 'total_connections': 15, 'total_tags': 30}
```

---

### `count() -> int`

Get total number of records.

---

## Two-Tier Memory API

Aura organizes memory into two logical tiers:

| Tier | Levels | Decay | Purpose |
|------|--------|-------|---------|
| **Cognitive** | Working + Decisions | 0.80–0.90/day | Ephemeral working memory — session notes, recent decisions, temporary context |
| **Core** | Domain + Identity | 0.95–0.99/day | Permanent knowledge base — facts, profile, domain expertise |

### `recall_cognitive(query=None, limit=None) -> list[Record]`

Search only the cognitive tier (WORKING + DECISIONS). Returns records sorted by importance.

```python
# All cognitive records
recent = brain.recall_cognitive()

# Search within cognitive tier
notes = brain.recall_cognitive("meeting notes", limit=10)
```

### `recall_core_tier(query=None, limit=None) -> list[Record]`

Search only the core tier (DOMAIN + IDENTITY). Returns records sorted by importance.

```python
# All core knowledge
knowledge = brain.recall_core_tier()

# Search within core tier
facts = brain.recall_core_tier("user preferences", limit=10)
```

### `tier_stats() -> dict`

Get memory statistics broken down by tier.

```python
stats = brain.tier_stats()
# {'cognitive_total': 15, 'cognitive_working': 10, 'cognitive_decisions': 5,
#  'core_total': 27, 'core_domain': 20, 'core_identity': 7,
#  'total': 42}
```

### `promotion_candidates(min_activations=None, min_strength=None) -> list[Record]`

Find cognitive records that qualify for promotion to core. A record qualifies when it has been recalled frequently (activation_count >= threshold) and maintains high strength.

```python
candidates = brain.promotion_candidates()
for rec in candidates:
    print(f"  {rec.content[:50]}  (activations={rec.activation_count})")
    brain.promote_record(rec.id)
```

### `promote_record(record_id) -> Level | None`

Manually promote a record to the next level. Returns the new level, or None if already at IDENTITY.

```python
# WORKING → DECISIONS → DOMAIN → IDENTITY
new_level = brain.promote_record(record_id)
```

### `Level` tier properties

```python
Level.Working.tier        # "cognitive"
Level.Decisions.tier      # "cognitive"
Level.Domain.tier         # "core"
Level.Identity.tier       # "core"

Level.Working.is_cognitive  # True
Level.Domain.is_core        # True
```

---

### `close()`

Flush and close the brain. Called automatically on garbage collection.

---

### `flush()`

Force flush all pending writes to disk.

---

### `is_encrypted() -> bool`

Check if the brain uses encryption.

---

## Import / Export

### `export_json() -> str`

Export all records as a JSON string.

### `import_json(json_str)`

Import records from a JSON string. Merges with existing data.

---

## Trust & Taxonomy

### `set_taxonomy(taxonomy: TagTaxonomy)`

Configure the tag taxonomy. Determines which tags are identity, stable, volatile, sensitive, etc.

```python
from aura import TagTaxonomy

tax = TagTaxonomy()
tax.identity_tags = {"name", "role", "language"}
tax.stable_tags = {"preference", "workflow"}
tax.volatile_tags = {"task", "temp"}
tax.sensitive_tags = {"credential", "financial", "contact"}
brain.set_taxonomy(tax)
```

### `get_taxonomy() -> TagTaxonomy`

Get current taxonomy configuration.

### `set_trust_config(config: TrustConfig)`

Configure trust scoring. Affects how recall results are re-ranked.

```python
from aura import TrustConfig

tc = TrustConfig()
tc.source_trust = {"user": 1.0, "api": 0.8, "agent": 0.7, "web_scrape": 0.5}
tc.source_authority = {"nature.com": 0.95, "wikipedia.org": 0.7}
tc.recency_boost_max = 0.1
tc.recency_half_life_days = 7.0
brain.set_trust_config(tc)
```

**Trust formula:** `effective_trust = (base_trust + recency_boost) * authority_multiplier`

---

## Identity

### `store_user_profile(fields: dict[str, str]) -> str`

Store or update user profile fields. Returns the record ID.

```python
brain.store_user_profile({"name": "Teo", "language": "Ukrainian", "role": "developer"})
```

### `get_user_profile() -> dict | None`

Get the current user profile as a dict.

### `set_persona(persona: AgentPersona) -> str`

Set the agent's persona. Returns the record ID.

```python
from aura import AgentPersona, PersonaTraits

persona = AgentPersona()
persona.name = "Atlas"
persona.role = "Research Assistant"
persona.traits = PersonaTraits()
brain.set_persona(persona)
```

### `get_persona() -> AgentPersona | None`

Get the current agent persona.

---

## Living Memory (Maintenance)

### `run_maintenance() -> MaintenanceReport`

Run a full 8-phase maintenance cycle:

1. **Level Fix** — Correct invalid level assignments
2. **Decay** — Apply level-specific retention rates, archive weak records
3. **Guarded Reflect** — Promote/demote records based on access patterns
4. **Insights** — Detect patterns, conflicts, clusters
5. **Consolidation** — Merge duplicates (MinHash, 85%+ similarity)
6. **Cross-Connections** — Discover 2-hop relationships in knowledge graph
7. **Scheduled Tasks** — Execute pending tasks and reminders
8. **Archival** — Archive records matching archival rules

```python
report = brain.run_maintenance()
print(f"Decayed: {report.decay.decayed}")
print(f"Promoted: {report.reflect.promoted}")
print(f"Merged: {report.consolidation.native_merged}")
print(f"Archived: {report.records_archived}")
print(f"Insights: {report.insights_found}")
print(f"Cross-connections: {report.cross_connections}")
```

### `configure_maintenance(config: MaintenanceConfig)`

Configure maintenance behavior.

```python
from aura import MaintenanceConfig

config = MaintenanceConfig()
config.decay_enabled = True
config.consolidation_enabled = True
config.synthesis_enabled = False
config.max_clusters_per_run = 100
brain.configure_maintenance(config)
```

### `start_background(interval_secs=120)`

Start background maintenance thread. Runs maintenance cycle at the specified interval.

### `stop_background()`

Stop the background maintenance thread.

### `is_background_running() -> bool`

Check if background maintenance is running.

---

## Research Orchestrator

### `start_research(topic, depth=None) -> dict`

Start a new research project. Returns a dict with `id`, `topic`, `depth`, `queries`.

```python
project = brain.start_research("Quantum Computing", depth="deep")
project_id = project["id"]
```

### `add_research_finding(project_id, query, result, url=None)`

Add a research finding to an active project.

```python
brain.add_research_finding(project_id,
    "What is quantum entanglement?",
    "Entanglement is a quantum phenomenon where particles become correlated",
    "https://nature.com/article")
```

### `complete_research(project_id, synthesis=None) -> str`

Complete a research project and generate a synthesis record. Returns the synthesis record ID.

```python
synthesis_id = brain.complete_research(project_id)
# Or provide custom synthesis:
synthesis_id = brain.complete_research(project_id, synthesis="Custom summary...")
```

### `active_research() -> list`

List all active (incomplete) research projects.

---

## Circuit Breaker

### `record_tool_failure(tool_name)`

Record a failure for a tool. After threshold failures, the circuit opens.

### `record_tool_success(tool_name)`

Record a success for a tool. Resets failure counter.

### `is_tool_available(tool_name) -> bool`

Check if a tool's circuit is closed (available). Returns False if the circuit is open.

### `tool_health(tool_name) -> str`

Get tool circuit state: `"closed"` (healthy), `"open"` (broken), or `"half_open"` (testing).

---

## Credibility

### `get_credibility(domain) -> float`

Get the credibility score for a domain (0.0–1.0). Includes 60+ pre-scored domains.

```python
brain.get_credibility("nature.com")       # 0.95
brain.get_credibility("reddit.com")       # 0.45
brain.get_credibility("unknown.com")      # 0.5 (default)
```

### `set_credibility_override(domain, score)`

Override the credibility score for a domain.

```python
brain.set_credibility_override("internal-wiki.com", 0.9)
```

---

## Individual Maintenance Operations

These are exposed for fine-grained control. Normally, `run_maintenance()` calls them all.

### `decay() -> DecayReport`

Run decay phase only.

### `reflect() -> ReflectReport`

Run reflect phase only (promote/demote based on access patterns).

### `consolidate() -> ConsolidationReport`

Run consolidation phase only (merge duplicates).

---

## Session Management

### `process(content, session_id=None) -> str`

Process content in a session context. Stores the content and returns the record ID.

### `end_session(session_id)`

End a session and clean up session state.

---

## Synonyms

### `load_synonyms(path)`

Load synonym definitions from a JSON file.

### `has_synonyms() -> bool`

Check if synonyms are loaded.

---

## Enums

### `Level`

Memory importance levels with two-tier grouping.

- `Level.Working` — Short-term memory (hours). Retention: 0.80/cycle. **Cognitive tier.**
- `Level.Decisions` — Medium-term (days). Retention: 0.90/cycle. **Cognitive tier.**
- `Level.Domain` — Long-term (weeks). Retention: 0.95/cycle. **Core tier.**
- `Level.Identity` — Near-permanent (months+). Retention: 0.99/cycle. **Core tier.**

**Properties:**
- `tier` (str) — `"cognitive"` or `"core"`
- `is_cognitive` (bool) — True for Working/Decisions
- `is_core` (bool) — True for Domain/Identity
- `decay_rate` (float) — Daily decay multiplier
- `dna` (str) — SDR classification (`"general"` or `"user_core"`)

---

## Data Classes

### `Record`

A memory record.

**Attributes:**
- `id` (str) — Unique identifier.
- `content` (str) — Text content.
- `level` (str) — Memory level.
- `strength` (float) — Current strength (0.0–1.0).
- `tags` (list[str]) — Tags.
- `connections` (dict[str, float]) — Bidirectional connections to other records (id → weight).
- `connection_types` (dict[str, str]) — Relationship types for connections (id → type). Types: `"causal"`, `"reflective"`, `"associative"`, `"coactivation"`, or custom.
- `metadata` (dict[str, str]) — Key-value metadata (includes provenance fields).
- `importance` (float) — Composite importance score (0.0–1.0). Formula: strength(40%) + level(25%) + connections(20%) + activations(15%).
- `created_at` (float) — Unix timestamp.
- `last_activated` (float) — Unix timestamp of last recall.

### `TagTaxonomy`

Configurable tag sets.

**Attributes:**
- `identity_tags` (set[str]) — Tags that mark identity-level records.
- `stable_tags` (set[str]) — Tags for stable, long-lived records.
- `volatile_tags` (set[str]) — Tags for short-lived records.
- `sensitive_tags` (set[str]) — Tags for sensitive data (triggers guards).
- `non_identity_tags` (set[str]) — Tags that should never be identity-level.
- `consolidation_skip_tags` (set[str]) — Tags to skip during consolidation.
- `archive_protected_tags` (set[str]) — Tags that prevent archival.

### `TrustConfig`

Source trust configuration.

**Attributes:**
- `source_trust` (dict[str, float]) — Trust scores per source channel.
- `source_authority` (dict[str, float]) — Authority multipliers per domain.
- `recency_boost_max` (float) — Maximum recency boost (default: 0.1).
- `recency_half_life_days` (float) — Half-life for recency boost (default: 7.0).

### `MaintenanceConfig`

Maintenance cycle configuration.

**Attributes:**
- `decay_enabled` (bool) — Enable decay phase.
- `consolidation_enabled` (bool) — Enable consolidation phase.
- `synthesis_enabled` (bool) — Enable knowledge synthesis.
- `max_clusters_per_run` (int) — Max clusters to process per cycle.

### `MaintenanceReport`

Report from a maintenance cycle.

**Attributes:**
- `timestamp` (str) — ISO timestamp.
- `total_records` (int) — Total records after maintenance.
- `decay` (DecayReport) — Decay phase results.
- `reflect` (ReflectReport) — Reflect phase results.
- `consolidation` (ConsolidationReport) — Consolidation results.
- `insights_found` (int) — Number of insights discovered.
- `cross_connections` (int) — Number of new cross-connections.
- `records_archived` (int) — Number of records archived.
- `task_reminders` (list[str]) — Triggered task reminders.

### `AgentPersona`

Agent persona configuration.

**Attributes:**
- `name` (str) — Agent name.
- `role` (str) — Agent role.
- `traits` (PersonaTraits) — Personality traits.

### `CircuitBreakerConfig`

Circuit breaker configuration.

**Attributes:**
- `failure_threshold` (int) — Failures before opening circuit.
- `recovery_timeout_secs` (int) — Seconds before attempting recovery.
