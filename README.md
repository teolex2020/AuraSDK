# AuraSDK: The Cognitive Memory Layer for Autonomous Agents

[![CI](https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml/badge.svg)](https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/aura-memory.svg)](https://pypi.org/project/aura-memory/)
[![Downloads](https://img.shields.io/pypi/dm/aura-memory.svg)](https://pypi.org/project/aura-memory/)
[![GitHub stars](https://img.shields.io/github/stars/teolex2020/AuraSDK?style=social)](https://github.com/teolex2020/AuraSDK/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Patent Pending](https://img.shields.io/badge/Patent_Pending-US_63%2F969%2C703-blue.svg)](https://www.uspto.gov/)

## The Problem

**LLMs forget everything.** Every conversation starts from zero. Standard vector databases store text but lack association, decay, and trust. Existing memory solutions (Mem0, Zep, Cognee) require LLM calls for basic recall — adding latency, cloud dependency, and cost to every operation.

**Your agent needs a long-term brain, not another API call.**

## The Solution

Aura gives your AI agent persistent, hierarchical memory that decays, consolidates, and evolves — like a human brain. No LLM calls. No embedding API. No cloud. One `pip install`, ~3 MB binary, works fully offline.

```bash
pip install aura-memory
```

```python
from aura import Aura, Level

brain = Aura("./agent_memory")

brain.store("User prefers dark mode", level=Level.Identity, tags=["ui"])
brain.store("Deploy to staging first", level=Level.Decisions, tags=["workflow"])

context = brain.recall("user preferences")  # <1ms, inject into any LLM
```

That's it. Your agent now remembers.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb)
[![Demo Video](https://img.shields.io/badge/YouTube-Demo_30s-red?logo=youtube)](https://www.youtube.com/watch?v=ZyE9P2_uKxg)

## Why Aura?

| | Aura | Mem0 | Zep | Cognee | Letta/MemGPT |
|---|---|---|---|---|---|
| **LLM required** | **No** | Yes | Yes | Yes (cognify) | Yes |
| **Recall latency** | **<1ms** | ~200ms+ | ~200ms | LLM-bound | LLM-bound |
| **Works offline** | **Fully** | Partial | No | No (needs API) | With local LLM |
| **Cost per operation** | **$0** | API billing | Credit-based | LLM + DB cost | LLM cost |
| **Binary size** | **~3 MB** | Python pkg | Cloud service | Heavy (Neo4j + vector DB) | Python pkg |
| **Memory decay & promotion** | **Built-in** | Via LLM | Via LLM | No | Via LLM |
| **Trust & provenance** | **Built-in** | No | No | No | No |
| **Encryption at rest** | **ChaCha20 + Argon2** | No | No | No | No |
| **Language** | **Rust** | Python | Proprietary | Python | Python |

## How Memory Works

Aura organizes memories into 4 levels across 2 tiers — important memories persist, trivial ones decay naturally:

```
CORE TIER (slow decay, weeks to months)
  Identity  [0.99]  Who the user is. Preferences. Personality.
  Domain    [0.95]  Learned facts. Domain knowledge.

COGNITIVE TIER (fast decay, hours to days)
  Decisions [0.90]  Choices made. Action items.
  Working   [0.80]  Current tasks. Recent messages.
```

One call runs the full lifecycle — decay, promote, merge duplicates, archive expired:

```python
report = brain.run_maintenance()  # 8 phases, <1ms
```

## Key Features

- **RRF Fusion Recall** — Multi-signal ranking: SDR + MinHash + Tag Jaccard (+ optional embeddings)
- **Trust & Provenance** — Source authority scoring: user input outranks web scrapes, automatically
- **Source Type Tracking** — Every memory carries provenance: `recorded`, `retrieved`, `inferred`, `generated`
- **Two-Tier Memory** — Cognitive (ephemeral) + Core (permanent) with dedicated recall and promotion API
- **Namespace Isolation** — `namespace="sandbox"` keeps test data invisible to production recall
- **Pluggable Embeddings** — Optional 4th RRF signal: bring your own embedding function
- **Auto-Protect Guards** — Detects phone numbers, emails, wallets, API keys automatically
- **Typed Connections** — Causal, reflective, associative graph links between memories
- **Background Maintenance** — 8-phase cycle: decay, reflect, insights, consolidation, archival
- **Encryption** — ChaCha20-Poly1305 with Argon2id key derivation
- **Pure Rust** — No Python dependencies, no external services, ~3 MB binary

## Quick Start

### Store & Recall

```python
from aura import Aura, Level

brain = Aura("./data")

# Store memories at different importance levels
brain.store("User prefers dark mode", level=Level.Identity, tags=["preference"])
brain.store("Deploy to staging first", level=Level.Decisions, tags=["workflow"])
brain.store("Bug in auth module", level=Level.Working, tags=["bug", "auth"])

# Recall returns formatted context — inject straight into LLM system prompt
context = brain.recall("authentication issues", token_budget=2000)

# Structured recall returns scored results with metadata
results = brain.recall_structured("user preferences", top_k=5)
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} -- {r['content'][:60]}")
```

### Trust & Provenance

```python
from aura import Aura, TrustConfig

brain = Aura("./data")

tc = TrustConfig()
tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
brain.set_trust_config(tc)

# User facts rank higher than scraped data in recall
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

## Performance

Benchmarked on Windows 10 / Ryzen 7 / 1,000 records:

| Operation | Latency | vs Mem0* |
|-----------|---------|----------|
| Store | 0.09 ms/op | ~same |
| Recall (structured) | 0.74 ms/op | **~270x faster** |
| Recall (cached) | 0.48 us/op | **~400,000x faster** |
| Search by tag | 0.01 ms/op | N/A |
| Maintenance cycle | 1.1 ms | No equivalent |
| Binary size | ~3 MB | ~50 MB+ (Python + deps) |

\*Mem0 recall requires embedding API call (~200ms+) + vector search. Aura recall is pure local computation.

## Dashboard UI

Aura includes a standalone web dashboard for visual memory management. Download from [GitHub Releases](https://github.com/teolex2020/AuraSDK/releases).

```bash
./aura-dashboard ./my_brain --port 8000
```

- **Analytics** — total memories, plasticity stats, DNA distribution
- **Memory Explorer** — paginated table with filtering, edit, delete, batch operations
- **Recall Console** — test RRF Fusion search with live scoring
- **Ingest** — add single or batch memories

| Platform | Binary |
|----------|--------|
| Windows x64 | `aura-dashboard-windows-x64.exe` |
| Linux x64 | `aura-dashboard-linux-x64` |
| macOS ARM | `aura-dashboard-macos-arm64` |
| macOS x64 | `aura-dashboard-macos-x64` |

## MCP Server (Claude Desktop)

Give Claude persistent memory across conversations:

```bash
pip install aura-memory
```

Add to Claude Desktop config (Settings > Developer > Edit Config):

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

## Examples

**Try now:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb) — zero install, runs in browser

**Cookbook:**
- [`personal_assistant.py`](examples/personal_assistant.py) — Agent that remembers preferences after a week offline

**Core:**
- [`basic_usage.py`](examples/basic_usage.py) — Store, recall, search, update, delete
- [`encryption.py`](examples/encryption.py) — Encrypted memory with ChaCha20-Poly1305
- [`agent_memory.py`](examples/agent_memory.py) — Trust, provenance, research, circuit breaker
- [`edge_device.py`](examples/edge_device.py) — IoT/edge: encryption, auto-protect, benchmarks
- [`maintenance_daemon.py`](examples/maintenance_daemon.py) — Background maintenance thread

**Integrations:**
- [`ollama_agent.py`](examples/ollama_agent.py) — Fully local AI assistant (Ollama, no API key)
- [`langchain_agent.py`](examples/langchain_agent.py) — LangChain: system prompt injection
- [`openai_agents.py`](examples/openai_agents.py) — OpenAI Agents SDK: dynamic instructions
- [`crewai_agent.py`](examples/crewai_agent.py) — CrewAI: tool-based recall/store
- [`autogen_agent.py`](examples/autogen_agent.py) — AutoGen: Memory protocol implementation
- [`research_bot.py`](examples/research_bot.py) — Research orchestrator with trust scoring

## Architecture

```
Python  --  from aura import Aura  -->  aura._core (PyO3)
                                              |
Rust    ------------------------------------------
        +---------------------------------------------+
        |  Aura (orchestrator)                         |
        |                                              |
        |  Two-Tier Memory                             |
        |  +-- Cognitive Tier (Working + Decisions)    |
        |  +-- Core Tier (Domain + Identity)           |
        |                                              |
        |  Recall Engine (RRF Fusion, k=60)            |
        |  +-- SDR similarity (256k bit)               |
        |  +-- MinHash N-gram                          |
        |  +-- Tag Jaccard                             |
        |  +-- Embedding (optional, pluggable)         |
        |                                              |
        |  Knowledge Graph (typed connections)          |
        |  Living Memory (8-phase maintenance)          |
        |  Trust & Provenance                           |
        |  Guards (auto-protect PII)                    |
        |  Encryption (ChaCha20 + Argon2id)             |
        +---------------------------------------------+
```

## API Reference

See [docs/API.md](docs/API.md) for the complete API reference (40+ methods).

## Resources

- [Demo Video (30s)](https://www.youtube.com/watch?v=ZyE9P2_uKxg) — Quick overview
- [API Reference](docs/API.md) — Complete API docs
- [Examples](examples/) — Ready-to-run scripts
- [Landing Page](https://aurasdk.dev) — Project overview

## Contributing

Contributions welcome! Check the [open issues](https://github.com/teolex2020/AuraSDK/issues) or open a new one.

If Aura is useful to you, a [GitHub star](https://github.com/teolex2020/AuraSDK) helps others discover it.

## License & Intellectual Property

- **Code License:** MIT — see [LICENSE](LICENSE).
- **Patent Notice:** The core cognitive architecture of Aura (DNA Layering, Cognitive Crystallization, SDR Indexing, and Synaptic Synthesis) is **Patent Pending** (US Provisional Application No. **63/969,703**). See [PATENT](PATENT) for details. Commercial integration of these architectural concepts into enterprise products requires a commercial license. The open-source SDK is freely available for non-commercial, academic, and standard agent integrations under the MIT License.

---

Built in Kyiv, Ukraine :ukraine: — including during power outages.
