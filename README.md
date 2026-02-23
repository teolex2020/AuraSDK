# Aura

[![CI](https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml/badge.svg)](https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/aura-memory.svg)](https://pypi.org/project/aura-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Patent Pending](https://img.shields.io/badge/Patent_Pending-US_63%2F969%2C703-blue.svg)](https://www.uspto.gov/)

Cognitive memory for AI agents. Pure Rust, no embeddings required.

Aura gives your AI agent persistent, hierarchical memory that decays, consolidates, and evolves — like human memory. No LLM calls. No embedding API. No cloud. One `pip install`, 2.7 MB binary, works offline.

```python
from aura import Aura, Level

brain = Aura("./my_agent")

# Store memories at different importance levels
brain.store("User prefers dark mode", level=Level.Identity, tags=["preference"])
brain.store("Deploy to staging first", level=Level.Decisions, tags=["workflow"])
brain.store("Bug in auth module", level=Level.Working, tags=["bug", "auth"])

# Recall with RRF Fusion (SDR + MinHash + Tag Jaccard)
context = brain.recall("authentication issues", token_budget=2000)

# Structured recall with trust-weighted scores
results = brain.recall_structured("user preferences", top_k=5)
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} — {r['content'][:60]}")

# Run maintenance (decay, consolidation, insights, archival)
report = brain.run_maintenance()
```

## Why Aura?

| | Aura | Mem0 | Zep | Letta/MemGPT |
|---|---|---|---|---|
| LLM required | **No** | Yes | Yes | Yes |
| Embedding model required | **No** | Yes | Yes | No |
| Works fully offline | **Yes** | Partial | No | With local LLM |
| Cost per operation | **$0** | API billing | Credit-based | LLM cost |
| Recall latency (1K records) | **<1ms** | ~200ms+ | ~200ms | LLM-bound |
| Binary size | **2.7 MB** | Python pkg | Cloud service | Python pkg |
| Memory lifecycle (decay/promote) | **Built-in** | Via LLM | Via LLM | Via LLM |
| Trust & provenance scoring | **Built-in** | No | No | No |
| Background maintenance (8 phases) | **Built-in** | No | No | No |
| Encryption at rest | **ChaCha20** | No | No | No |
| Language | **Rust** | Python | Proprietary | Python |

## Features

- **RRF Fusion Recall** — Multi-signal ranking: SDR + MinHash + Tag Jaccard (+ optional embeddings), k=60
- **4-Level Hierarchical Decay** — Working (0.80), Decisions (0.90), Domain (0.95), Identity (0.99) retention rates
- **Living Memory** — 8-phase background maintenance: decay, reflect, insights, consolidation, cross-connections, archival
- **Trust & Provenance** — Source authority scoring, provenance stamping, credibility tracking for 60+ domains
- **Typed Connections** — Causal, reflective, associative, coactivation graph links between memories
- **Auto-Protect Guards** — Regex detection of phone numbers, emails, wallets, API keys with automatic tagging
- **Pluggable Embeddings** — Optional 4th RRF signal: bring your own embedding function for semantic boost
- **Research Orchestrator** — Start research sessions, collect findings with credibility, synthesize results
- **Circuit Breaker** — Per-tool failure tracking with automatic circuit open/close
- **Encryption** — ChaCha20-Poly1305 with Argon2id key derivation (optional)
- **Pure Rust** — No Python dependencies, no external services, <3 MB binary

## Installation

```bash
pip install aura-memory
```

Requires Python 3.9+. Pre-built wheels for Linux, macOS, and Windows.

## Quick Start

### Basic Store & Recall

```python
from aura import Aura, Level

brain = Aura("./data")

# Store with auto-deduplication
brain.store("Python uses indentation for blocks", tags=["python", "syntax"])
brain.store("Rust has zero-cost abstractions", level=Level.Domain, tags=["rust"])

# Recall returns formatted context for LLM injection
context = brain.recall("programming language features")
print(context)
# === COGNITIVE CONTEXT ===
# [DOMAIN]
#   - Rust has zero-cost abstractions [rust]
# [WORKING]
#   - Python uses indentation for blocks [python, syntax]

# Structured recall returns scored results
results = brain.recall_structured("rust", top_k=3)
for r in results:
    print(f"  score={r['score']:.3f}  {r['content']}")
```

### Pluggable Embeddings (Optional)

Aura works without embeddings, but you can add them as a 4th RRF signal for better semantic recall:

```python
from aura import Aura

brain = Aura("./data")

# Plug in any embedding function: OpenAI, Ollama, sentence-transformers, etc.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

brain.set_embedding_fn(lambda text: model.encode(text).tolist())

# Now store and recall use embeddings as a 4th signal in RRF fusion
brain.store("Authentication failed for user admin")
brain.store("Login issues reported on the dashboard")

# "login problems" matches via embedding even without exact word overlap
results = brain.recall_structured("login problems", top_k=5)
```

When offline or without an embedding model, Aura falls back to SDR + MinHash + Tag Jaccard — still fast, still effective.

### Trust & Provenance

```python
from aura import Aura, TrustConfig

brain = Aura("./data")

# Configure source trust
tc = TrustConfig()
tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
brain.set_trust_config(tc)

# Store with provenance (channel stamps source automatically)
brain.store("API response data", tags=["api"], channel="api_endpoint")

# Structured recall applies trust weighting
# final_score = rrf_score * effective_trust
results = brain.recall_structured("api data", top_k=5)
```

### Living Memory (Background Maintenance)

```python
from aura import Aura, MaintenanceConfig

brain = Aura("./data")

# Single maintenance cycle (8 phases)
report = brain.run_maintenance()
print(f"Decayed: {report.decay.decayed}")
print(f"Promoted: {report.reflect.promoted}")
print(f"Consolidated: {report.consolidation.native_merged}")
print(f"Archived: {report.records_archived}")

# Start background maintenance thread
brain.start_background(interval_secs=120)
```

### Encryption

```python
# All data encrypted at rest with ChaCha20-Poly1305
brain = Aura("./secret_data", password="my-secure-password")
brain.store("Top secret information")
assert brain.is_encrypted()
```

## Use Cases

### Aura + Ollama (fully local AI with memory)

```python
# No cloud. No API keys. Everything runs on your machine.
# See examples/ollama_agent.py for the full example.

brain = Aura("./agent_data")
context = brain.recall(user_message, token_budget=2000)
# Inject context into Ollama system prompt...
```

### Edge / IoT / Air-gapped

```python
# 2.7 MB binary, encrypted, zero network dependencies
brain = Aura("./edge_data", password="device-key")
brain.store("Sensor reading: 22.5C", tags=["sensor"])
# Auto-protect detects PII: phone, email, wallet, API key
brain.store("User phone: +380991234567", tags=["user"])
# -> automatically adds "contact" tag, blocks consolidation
```

## Performance

Benchmarked on Windows 10 / Ryzen 7 / 1000 records:

| Operation | Latency | vs Mem0* |
|-----------|---------|----------|
| Store | 0.09 ms/op | ~same |
| Recall (structured) | 0.74 ms/op | **~270x faster** |
| Recall (cached) | 0.48 us/op | **~400,000x faster** |
| Search by tag | 0.01 ms/op | N/A |
| Maintenance cycle | 1.1 ms | No equivalent |
| Binary size | 2.7 MB | ~50 MB+ (Python + deps) |

*Mem0 recall requires embedding API call (~200ms+) + vector search. Aura recall is pure local computation.

## Memory Levels

| Level | Retention | Use Case |
|-------|-----------|----------|
| `Level.Working` | 0.80/cycle | Short-term: current tasks, recent messages |
| `Level.Decisions` | 0.90/cycle | Medium-term: choices made, action items |
| `Level.Domain` | 0.95/cycle | Long-term: learned facts, domain knowledge |
| `Level.Identity` | 0.99/cycle | Permanent: user preferences, core identity |

Records are automatically promoted based on access patterns and demoted through decay.

## Architecture

```
Python  --  from aura import Aura  -->  aura._core (PyO3)
                                              |
Rust    ------------------------------------------
        +---------------------------------------------+
        |  Aura (orchestrator)                         |
        |  +-- SDR Engine (256k bits, xxHash3)         |
        |  +-- RRF Fusion Recall (k=60)                |
        |      +-- Signal 1: SDR similarity            |
        |      +-- Signal 2: MinHash N-gram            |
        |      +-- Signal 3: Tag Jaccard               |
        |      +-- Signal 4: Embedding (optional)      |
        |  +-- Knowledge Graph (typed connections)      |
        |  +-- Living Memory (8-phase maintenance)      |
        |  +-- Trust & Provenance                       |
        |  +-- Guards (auto-protect sensitive data)     |
        |  +-- Circuit Breaker (per-tool)               |
        |  +-- Research Orchestrator                    |
        |  +-- Identity (profile + persona)             |
        |  +-- Encryption (ChaCha20 + Argon2id)         |
        +---------------------------------------------+
```

## Examples

- [`basic_usage.py`](examples/basic_usage.py) — Store, recall, search, update, delete
- [`ollama_agent.py`](examples/ollama_agent.py) — Fully local AI assistant with persistent memory
- [`research_bot.py`](examples/research_bot.py) — Research orchestrator with trust scoring
- [`edge_device.py`](examples/edge_device.py) — IoT/edge: encryption, auto-protect, circuit breaker

## API Reference

See [docs/API.md](docs/API.md) for the complete API reference (40+ methods).

## License & Intellectual Property

- **Code License:** MIT — see [LICENSE](LICENSE).
- **Patent Notice:** The core cognitive architecture of Aura (DNA Layering, Cognitive Crystallization, SDR Indexing, and Synaptic Synthesis) is **Patent Pending** (US Provisional Application No. **63/969,703**). See [PATENT](PATENT) for details. Commercial integration of these architectural concepts into enterprise products requires a commercial license. The open-source SDK is freely available for non-commercial, academic, and standard agent integrations under the MIT License.
