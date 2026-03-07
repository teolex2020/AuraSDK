<p align="center">
  <h1 align="center">AuraSDK</h1>
  <p align="center"><strong>Cognitive Memory Engine for AI Agents</strong></p>
  <p align="center">
    Learns from experience · <1ms recall · No LLM calls · No cloud · ~3 MB
  </p>
</p>

<p align="center">
  <a href="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml"><img src="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/aura-memory/"><img src="https://img.shields.io/pypi/v/aura-memory.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/aura-memory/"><img src="https://img.shields.io/pypi/dm/aura-memory.svg" alt="Downloads"></a>
  <a href="https://github.com/teolex2020/AuraSDK/stargazers"><img src="https://img.shields.io/github/stars/teolex2020/AuraSDK?style=social" alt="GitHub stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml"><img src="https://img.shields.io/badge/tests-619_passed-brightgreen" alt="Tests"></a>
  <a href="https://www.uspto.gov/"><img src="https://img.shields.io/badge/Patent_Pending-US_63%2F969%2C703-blue.svg" alt="Patent Pending"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>&nbsp;&nbsp;
  <a href="https://www.youtube.com/watch?v=ZyE9P2_uKxg"><img src="https://img.shields.io/badge/YouTube-Demo_30s-red?logo=youtube" alt="Demo Video"></a>&nbsp;&nbsp;
  <a href="https://aurasdk.dev"><img src="https://img.shields.io/badge/Web-aurasdk.dev-blue" alt="Website"></a>
</p>

---

LLMs forget everything. Every conversation starts from zero. Existing solutions bolt on vector databases and LLM calls for basic recall, adding latency, cloud dependency, and cost to every operation.

Aura gives your AI agent an adaptive cognitive layer: memory that decays, consolidates, learns from feedback, and evolves over time. Like a brain, not a database. One `pip install`, works fully offline.

```bash
pip install aura-memory
```

```python
from aura import Aura, Level

brain = Aura("./agent_memory")

brain.store("User prefers dark mode", level=Level.Identity, tags=["ui"])
brain.store("Deploy to staging first", level=Level.Decisions, tags=["workflow"])

context = brain.recall("user preferences")  # <1ms — inject into any LLM prompt
```

Your agent now remembers. No API keys. No embeddings. No config.

> **⭐ If AuraSDK is useful to you, a [GitHub star](https://github.com/teolex2020/AuraSDK) helps us get funding to continue development from Kyiv.**

---

## Why Aura?

| | **Aura** | Mem0 | Zep | Cognee | Letta/MemGPT |
|---|---|---|---|---|---|
| **Architecture** | **Cognitive engine** | Vector + LLM | Vector + LLM | Graph + LLM | LLM orchestration |
| **LLM required** | **No** | Yes | Yes | Yes | Yes |
| **Recall latency** | **<1ms** | ~200ms+ | ~200ms | LLM-bound | LLM-bound |
| **Works offline** | **Fully** | Partial | No | No | With local LLM |
| **Cost per operation** | **$0** | API billing | Credit-based | LLM + DB cost | LLM cost |
| **Binary size** | **~3 MB** | ~50 MB+ | Cloud service | Heavy (Neo4j+) | Python pkg |
| **Memory decay & promotion** | **Built-in** | Via LLM | Via LLM | No | Via LLM |
| **Trust & provenance** | **Built-in** | No | No | No | No |
| **Encryption at rest** | **ChaCha20 + Argon2** | No | No | No | No |
| **Language** | **Rust** | Python | Proprietary | Python | Python |

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

## How Memory Works

Aura organizes memories into 4 levels across 2 tiers. Important memories persist, trivial ones decay naturally:

```
CORE TIER (slow decay — weeks to months)
  Identity  [0.99]  Who the user is. Preferences. Personality.
  Domain    [0.95]  Learned facts. Domain knowledge.

COGNITIVE TIER (fast decay — hours to days)
  Decisions [0.90]  Choices made. Action items.
  Working   [0.80]  Current tasks. Recent context.
```

One call runs the full lifecycle — decay, promote, merge duplicates, archive expired:

```python
report = brain.run_maintenance()  # 8 phases, <1ms
```

---

## Key Features

**Core Memory Engine**
- **RRF Fusion Recall** — Multi-signal ranking: SDR + MinHash + Tag Jaccard (+ optional embeddings)
- **Two-Tier Memory** — Cognitive (ephemeral) + Core (permanent) with decay, promotion, and archival
- **Background Maintenance** — 8-phase lifecycle: decay, reflect, insights, consolidation, archival
- **Namespace Isolation** — `namespace="sandbox"` keeps test data invisible to production recall
- **Pluggable Embeddings** — Optional 4th RRF signal: bring your own embedding function

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

Without embeddings, Aura falls back to SDR + MinHash + Tag Jaccard — still fast, still effective.

### Encryption

```python
brain = Aura("./secret_data", password="my-secure-password")
brain.store("Top secret information")
assert brain.is_encrypted()  # ChaCha20-Poly1305 + Argon2id
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

## MCP Server (Claude Desktop)

Give Claude persistent memory across conversations:

```bash
pip install aura-memory
```

Add to Claude Desktop config (Settings → Developer → Edit Config):

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

Provides 8 tools: `recall`, `recall_structured`, `store`, `store_code`, `store_decision`, `search`, `insights`, `consolidate`.

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

52 Rust modules · ~23,500 lines · **272 Rust + 347 Python = 619 tests**

```
Python  ──  from aura import Aura  ──▶  aura._core (PyO3)
                                              │
Rust    ──────────────────────────────────────┘
        ┌─────────────────────────────────────────────┐
        │  Aura Engine                                │
        │                                             │
        │  Two-Tier Memory                            │
        │  ├── Cognitive Tier (Working + Decisions)   │
        │  └── Core Tier (Domain + Identity)          │
        │                                             │
        │  Recall Engine (RRF Fusion, k=60)           │
        │  ├── SDR similarity (256k bit)              │
        │  ├── MinHash N-gram                         │
        │  ├── Tag Jaccard                            │
        │  └── Embedding (optional, pluggable)        │
        │                                             │
        │  Adaptive Memory                            │
        │  ├── Feedback learning (boost/weaken)       │
        │  ├── Snapshots & rollback                   │
        │  ├── Supersede (version chains)             │
        │  └── Agent-to-agent sharing protocol        │
        │                                             │
        │  Knowledge Graph · Living Memory            │
        │  Trust & Provenance · PII Guards            │
        │  Encryption (ChaCha20 + Argon2id)           │
        │  StorageBackend (Fs / Memory / WASM)        │
        │  Telemetry (Prometheus + OpenTelemetry)      │
        └─────────────────────────────────────────────┘
```

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
- **Patent Notice:** The core cognitive architecture (DNA Layering, Cognitive Crystallization, SDR Indexing, Synaptic Synthesis) is **Patent Pending** (US Provisional Application No. **63/969,703**). See [PATENT](PATENT) for details. Commercial integration of these architectural concepts into enterprise products requires a commercial license. The open-source SDK is freely available under MIT for non-commercial, academic, and standard agent integrations.

---

<p align="center">
  Built in Kyiv, Ukraine 🇺🇦 — including during power outages.<br>
  <sub>Solo developer project. If you find this useful, your star means more than you think.</sub>
</p>

