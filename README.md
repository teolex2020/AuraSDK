<p align="center">
  <h1 align="center">AuraSDK</h1>
  <p align="center"><strong>Cognitive Layer That Makes Your AI Model Smarter Over Time</strong></p>
  <p align="center">
    Self-learning · No fine-tuning · No cloud training · <1ms recall · ~3 MB
  </p>
</p>

<p align="center">
  <a href="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml"><img src="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/aura-memory/"><img src="https://img.shields.io/pypi/v/aura-memory.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/aura-memory/"><img src="https://img.shields.io/pypi/dm/aura-memory.svg" alt="Downloads"></a>
  <a href="https://github.com/teolex2020/AuraSDK/stargazers"><img src="https://img.shields.io/github/stars/teolex2020/AuraSDK?style=social" alt="GitHub stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml"><img src="https://img.shields.io/badge/tests-828_passed-brightgreen" alt="Tests"></a>
  <a href="https://www.uspto.gov/"><img src="https://img.shields.io/badge/Patent_Pending-US_63%2F969%2C703-blue.svg" alt="Patent Pending"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>&nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=ZyE9P2_uKxg"><img src="https://img.shields.io/badge/YouTube-Demo_30s-red?logo=youtube" alt="Demo Video"></a>&nbsp;&nbsp;
  <a href="https://aurasdk.dev"><img src="https://img.shields.io/badge/Web-aurasdk.dev-blue" alt="Website"></a>
</p>

---

Your AI model is smart. But it forgets everything after every conversation — and it never gets smarter from experience.

AuraSDK is a cognitive layer that runs alongside any LLM. It observes every interaction, builds beliefs from patterns, discovers causal relationships, and derives behavioral policies — all locally, without fine-tuning or cloud training. The longer it runs, the smarter your agent becomes.

```bash
pip install aura-memory
```

```python
from aura import Aura, Level

brain = Aura("./agent_memory")
brain.enable_full_cognitive_stack()  # activate all 5 cognitive layers

# store what happens
brain.store("User always deploys to staging first", level=Level.Domain, tags=["workflow"])
brain.store("Staging deploy prevented 3 production incidents", level=Level.Domain, tags=["workflow"])

# recall — cognitive layer automatically surfaces relevant patterns
context = brain.recall("deployment decision")  # <1ms, no API call

# after enough interactions, the system derives this on its own:
hints = brain.get_surfaced_policy_hints()
# → [{"action": "Prefer", "domain": "workflow", "description": "deploy to staging first"}]
```

No API keys. No embeddings. No cloud. The model stays the same — your agent gets smarter.

> **⭐ If AuraSDK is useful to you, a [GitHub star](https://github.com/teolex2020/AuraSDK) helps us get funding to continue development from Kyiv.**

---

## Why Aura?

| | **Aura** | Mem0 | Zep | Cognee | Letta/MemGPT |
|---|---|---|---|---|---|
| **Architecture** | **5-layer cognitive engine** | Vector + LLM | Vector + LLM | Graph + LLM | LLM orchestration |
| **Self-learning without LLM** | **Yes — Belief→Causal→Policy** | No | No | No | No |
| **Behavioral policies from experience** | **Yes — automatic** | No | No | No | No |
| **LLM required** | **No** | Yes | Yes | Yes | Yes |
| **Recall latency** | **<1ms** | ~200ms+ | ~200ms | LLM-bound | LLM-bound |
| **Works offline** | **Fully** | Partial | No | No | With local LLM |
| **Cost per operation** | **$0** | API billing | Credit-based | LLM + DB cost | LLM cost |
| **Binary size** | **~3 MB** | ~50 MB+ | Cloud service | Heavy (Neo4j+) | Python pkg |
| **Memory decay & promotion** | **Built-in** | Via LLM | Via LLM | No | Via LLM |
| **Trust & provenance** | **Built-in** | No | No | No | No |
| **Encryption at rest** | **ChaCha20 + Argon2** | No | No | No | No |
| **Language** | **Rust** | Python | Proprietary | Python | Python |

### The Core Idea: Cheap Model + Aura > Expensive Model Alone

Fine-tuning costs thousands of dollars and weeks of work. RAG requires embeddings and a vector database. Context windows are expensive per token.

Aura gives you a third path: **a cognitive layer that accumulates experience between conversations** — free, local, sub-millisecond.

```
Week 1: GPT-4o-mini + Aura                Week 1: GPT-4 alone
  → average answers                          → average answers

Week 4: GPT-4o-mini + Aura                Week 4: GPT-4 alone
  → knows your workflow                      → still forgets everything
  → surfaces patterns you repeat             → same cost per token
  → warns before risky actions               → no improvement
  → $0 compute cost                          → still billing per call
```

The model stays the same. The agent gets smarter. That's Aura.

### Performance

Benchmarked on 1,000 records (Windows 10 / Ryzen 7):

| Operation | Latency | vs Mem0 |
|-----------|---------|---------|
| Store | 0.09 ms | ~same |
| Recall (structured) | 0.74 ms | **~270× faster** |
| Recall (cached) | 0.48 µs | **~400,000× faster** |
| Maintenance cycle | 1.1 ms | No equivalent |

Mem0 recall requires an embedding API call (~200ms+) + vector search. Aura recall is pure local computation.

---

## What Ships Today

Aura's full cognitive recall pipeline is active and bounded:

`Record → Belief (±5%) → Concept (±4%) → Causal (±3%) → Policy (±2%)`

Enable everything in one call:

```python
brain.enable_full_cognitive_stack()   # activates all four bounded reranking phases
brain.disable_full_cognitive_stack()  # back to raw RRF baseline
```

Or configure individual phases:

```python
brain.set_belief_rerank_mode("limited")   # belief-aware ranking
brain.set_concept_surface_mode("inspect") # concept annotations, no ranking change
brain.set_causal_rerank_mode("limited")   # causal chain boost
brain.set_policy_rerank_mode("limited")   # policy hint shaping
```

Higher layers also expose advisory surfaced output:

- `get_surfaced_concepts()` — stable concept abstractions over repeated beliefs
- `get_surfaced_causal_patterns()` — learned cause→effect patterns
- `get_surfaced_policy_hints()` — behavioral recommendations (Prefer / Avoid / Warn)
- no automatic behavior influence — all output is advisory and read-only

---

## How Memory Works

Aura organizes memories into 4 levels across 2 tiers. Important memories persist, trivial ones decay naturally:

```
CORE TIER (slow decay — weeks to months)
  Identity  [0.99]  Who the user is. Preferences. Personality.
  Domain    [0.95]  Learned facts. Domain knowledge.

COGNITIVE TIER (fast decay — hours to days)
  Decisions [0.90]  Choices made. Action items.
  Working   [0.80]  Current tasks. Recent context.

SEMANTIC TYPES (modulate decay & promotion)
  fact          Default knowledge record.
  decision      More persistent than a standard fact. Promotes earlier.
  preference    Long-lived user or agent preference.
  contradiction Preserved longer for conflict analysis.
  trend         Time-sensitive pattern tracked over repeated activation.
  serendipity   Cross-domain discovery record.
```

One call runs the lifecycle — decay, promotion, consolidation, and archival:

```python
report = brain.run_maintenance()  # background memory maintenance
```

---

## Key Features

**Core Memory Engine**
- **Fast Local Recall** - Multi-signal ranking with optional embedding support
- **Two-Tier Memory** — Cognitive (ephemeral) + Core (permanent) with decay, promotion, and archival
- **Semantic Memory Types** — 6 roles (`fact`, `decision`, `trend`, `preference`, `contradiction`, `serendipity`) that influence memory behavior and insighting
- **Phase-Based Insights** — Detects conflicts, trends, preference patterns, and cross-domain links
- **Background Maintenance** — Continuous memory hygiene: decay, reflect, insights, consolidation, archival
- **Namespace Isolation** — `namespace="sandbox"` keeps test data invisible to production recall
- **Pluggable Embeddings** - Optional embedding support: bring your own embedding function

**Trust & Safety**
- **Trust & Provenance** — Source authority scoring: user input outranks web scrapes, automatically
- **Source Type Tracking** — Every memory carries provenance: `recorded`, `retrieved`, `inferred`, `generated`
- **Auto-Protect Guards** — Detects phone numbers, emails, wallets, API keys automatically
- **Encryption** — ChaCha20-Poly1305 with Argon2id key derivation

**Adaptive Memory**
- **Feedback Learning** — `brain.feedback(id, useful=True)` boosts useful memories, weakens noise
- **Semantic Versioning** — `brain.supersede(old_id, new_content)` with full version chains
- **Snapshots & Rollback** — `brain.snapshot("v1")` / `brain.rollback("v1")` / `brain.diff("v1","v2")`
- **Agent-to-Agent Sharing** — `export_context()` / `import_context()` with trust metadata

**Enterprise & Integrations**
- **Multimodal Stubs** — `store_image()` / `store_audio_transcript()` with media provenance
- **Prometheus Metrics** — `/metrics` endpoint with 10+ business-level counters and histograms
- **OpenTelemetry** — `telemetry` feature flag with OTLP export and 17 instrumented spans
- **MCP Server** — Claude Desktop integration out of the box
- **WASM-Ready** — `StorageBackend` trait abstraction (`FsBackend` + `MemoryBackend`)
- **Pure Rust Core** — No Python dependencies, no external services

---

**Advisory Cognitive Overlays**
- **Belief-Aware Recall Rerank** — bounded production influence with strict guardrails
- **Concept Inspect Surface** — surfaced concepts and per-record annotations for inspection only
- **Causal / Policy Overlays** — advisory surfaced output only, no automatic control path

## Quick Start

### Trust & Provenance

```python
from aura import Aura, TrustConfig

brain = Aura("./data")

tc = TrustConfig()
tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
brain.set_trust_config(tc)

# User facts always rank higher than scraped data in recall
brain.store("User is vegan", channel="user")
brain.store("User might like steak restaurants", channel="web_scrape")

results = brain.recall_structured("food preferences", top_k=5)
# -> "User is vegan" scores higher, always
```

### Pluggable Embeddings (Optional)

```python
from aura import Aura

brain = Aura("./data")

# Plug in any embedding function: OpenAI, Ollama, sentence-transformers, etc.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
brain.set_embedding_fn(lambda text: model.encode(text).tolist())

# Now "login problems" matches "Authentication failed" via semantic similarity
brain.store("Authentication failed for user admin")
results = brain.recall_structured("login problems", top_k=5)
```

Without embeddings, Aura continues to use its local recall pipeline - still fast, still effective.

### Encryption

```python
brain = Aura("./secret_data", password="my-secure-password")
brain.store("Top secret information")
assert brain.is_encrypted()  # ChaCha20-Poly1305 + Argon2id
```

### Semantic Memory Types

```python
brain = Aura("./data")

# Decisions are treated as higher-value memory
brain.store("Use PostgreSQL over MySQL", semantic_type="decision", tags=["db"])

# Preferences persist longer than generic working notes
brain.store("User prefers dark mode", semantic_type="preference", tags=["ui"])

# Contradictions are preserved for conflict analysis
brain.store("User said vegan but ordered steak", semantic_type="contradiction")

# Search by semantic type
decisions = brain.search(semantic_type="decision")

# Cross-domain insights surface higher-level patterns
insights = brain.insights(phase=2)
# Example:
# [{'insight_type': 'preference_pattern', 'description': 'Preference cluster around ui', ...}]
```

### Namespace Isolation

```python
brain = Aura("./data")

brain.store("Real preference: dark mode", namespace="default")
brain.store("Test: user likes light mode", namespace="sandbox")

# Recall only sees "default" namespace — sandbox is invisible
results = brain.recall_structured("user preference", top_k=5)
```

---

## Cookbook: Personal Assistant That Remembers

The killer use case: an agent that remembers your preferences after a week offline, with zero API calls.

See [`examples/personal_assistant.py`](examples/personal_assistant.py) for the full runnable script.

```python
from aura import Aura, Level

brain = Aura("./assistant_memory")

# Day 1: User tells the agent about themselves
brain.store("User is vegan", level=Level.Identity, tags=["diet"])
brain.store("User loves jazz music", level=Level.Identity, tags=["music"])
brain.store("User works 10am-6pm", level=Level.Identity, tags=["schedule"])
brain.store("Discuss quarterly report tomorrow", level=Level.Working, tags=["task"])

# Simulate a week passing — run maintenance cycles
for _ in range(7):
    brain.run_maintenance()  # decay + reflect + consolidate + archive

# Day 8: What does the agent remember?
context = brain.recall("user preferences and personality")
# -> Still remembers: vegan, jazz, schedule (Identity, strength ~0.93)
# -> "quarterly report" decayed heavily (Working, strength ~0.21)
```

Identity persists. Tasks fade. Important patterns get promoted. Like a real brain.

---

## MCP Server — Claude Desktop · Cursor · Zed · VS Code

Give any MCP-compatible AI persistent, self-organizing memory:

```bash
pip install aura-memory
```

**Claude Desktop** — Settings → Developer → Edit Config:

```json
{
  "mcpServers": {
    "aura": {
      "command": "python",
      "args": ["-m", "aura", "mcp", "C:\\Users\\YOUR_NAME\\aura_brain"]
    }
  }
}
```

**Cursor / VS Code** — `.cursor/mcp.json` or `.vscode/mcp.json`:

```json
{
  "servers": {
    "aura": {
      "command": "python",
      "args": ["-m", "aura", "mcp", "./aura_brain"],
      "type": "stdio"
    }
  }
}
```

**macOS / Linux path:**
```bash
python -m aura mcp ~/aura_brain
```

Once connected, Claude automatically has 11 tools:

| Tool | Purpose |
|------|---------|
| `recall` | Retrieve relevant memories before answering |
| `recall_structured` | Get memories with scores and metadata |
| `store` | Save a fact, note, or context |
| `store_code` | Save a code snippet at Domain level |
| `store_decision` | Save a decision with reasoning |
| `search` | Filter memories by level or tags |
| `insights` | Memory health stats |
| `consolidate` | Merge similar records |
| `get` | Fetch a specific record by ID |
| `delete` | Remove a record by ID |
| `maintain` | Run a full maintenance cycle |

> After connecting, tell Claude: *"Before answering, always recall relevant context from memory. After our conversation, store key facts."*

---

## Dashboard UI

Aura includes a standalone web dashboard for visual memory management. Download from [GitHub Releases](https://github.com/teolex2020/AuraSDK/releases).

```bash
./aura-dashboard ./my_brain --port 8000
```

**Features:** Analytics · Memory Explorer with filtering · Recall Console with live scoring · Batch ingest

| Platform | Binary |
|----------|--------|
| Windows x64 | `aura-dashboard-windows-x64.exe` |
| Linux x64 | `aura-dashboard-linux-x64` |
| macOS ARM | `aura-dashboard-macos-arm64` |
| macOS x64 | `aura-dashboard-macos-x64` |

---

## Integrations & Examples

**Try now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb) — zero install, runs in browser

| Integration | Description | Link |
|-------------|-------------|------|
| Ollama | Fully local AI assistant, no API key needed | [`ollama_agent.py`](examples/ollama_agent.py) |
| LangChain | Drop-in Memory class + prompt injection | [`langchain_agent.py`](examples/langchain_agent.py) |
| LlamaIndex | Chat engine with persistent memory recall | [`llamaindex_agent.py`](examples/llamaindex_agent.py) |
| OpenAI Agents | Dynamic instructions with persistent memory | [`openai_agents.py`](examples/openai_agents.py) |
| Claude SDK | System prompt injection + tool use patterns | [`claude_sdk_agent.py`](examples/claude_sdk_agent.py) |
| CrewAI | Tool-based recall/store for crew agents | [`crewai_agent.py`](examples/crewai_agent.py) |
| AutoGen | Memory protocol implementation | [`autogen_agent.py`](examples/autogen_agent.py) |
| FastAPI | Per-user memory middleware with namespace isolation | [`fastapi_middleware.py`](examples/fastapi_middleware.py) |

**FFI (C/Go/C#):** [`aura.h`](examples/aura.h) · [`go/main.go`](examples/go/main.go) · [`csharp/Program.cs`](examples/csharp/Program.cs)

**More examples:** [`basic_usage.py`](examples/basic_usage.py) · [`encryption.py`](examples/encryption.py) · [`agent_memory.py`](examples/agent_memory.py) · [`edge_device.py`](examples/edge_device.py) · [`maintenance_daemon.py`](examples/maintenance_daemon.py) · [`research_bot.py`](examples/research_bot.py)

---

## Architecture

Aura uses a Rust core with Python bindings and a local-first memory runtime.

Publicly documented concepts are:

- Two-tier memory: cognitive + core
- Semantic roles for records
- Local multi-signal recall
- Belief-aware bounded reranking
- Trust, provenance, and namespace isolation
- Maintenance, insights, consolidation, and versioning

Higher cognitive layers may be present in the SDK as advisory inspection surfaces. They are not default runtime decision-making or behavior control.

The public repository documents the user-facing behavior and integration surface. Detailed internal architecture, tuning, and research notes are intentionally not published.

---

## Resources

- [Demo Video (30s)](https://www.youtube.com/watch?v=ZyE9P2_uKxg) — Quick overview
- [Examples](examples/) — Ready-to-run scripts
- [Landing Page](https://aurasdk.dev) — Project overview

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines, or check the [open issues](https://github.com/teolex2020/AuraSDK/issues).

⭐ **If Aura saves you time, a [GitHub star](https://github.com/teolex2020/AuraSDK) helps others discover it and helps us continue development.**

---

## License & Intellectual Property

- **Code License:** MIT — see [LICENSE](LICENSE).
- **Patent Notice:** Core architectural concepts are **Patent Pending** (US Provisional Application No. **63/969,703**). See [PATENT](PATENT) for details. The SDK source code is available under MIT. Separate commercial licensing is available for organizations that want contractual rights around patented architecture, OEM embedding, enterprise deployment, or dedicated support.
- **Commercial Licensing:** If you want to embed Aura's architecture into a commercial product, see [COMMERCIAL.md](COMMERCIAL.md).

---

<p align="center">
  Built in Kyiv, Ukraine 🇺🇦 — including during power outages.<br>
  <sub>Solo developer project. If you find this useful, your star means more than you think.</sub>
</p>


