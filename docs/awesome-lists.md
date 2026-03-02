# Awesome Lists Strategy for AuraSDK

## Summary

| List | Stars | Min Stars Required | Can Submit Now? | Section |
|------|-------|--------------------|-----------------|---------|
| awesome-rust | 47K | **50** | After 50 stars | Machine Learning |
| awesome-python | 228K | **100** (Hidden Gem) | After 100 stars | Machine Learning |
| Awesome-AI-Agents (Jenqyang) | ~1K | None stated | **Yes** | Tools |
| Awesome-Agent-Memory (TeleAI) | ~500 | None stated | **Yes** |  Open-Source Products |
| awesome-ai-sdks (e2b) | 1.1K | None stated | **Yes** | New section or SDK |
| Awesome-RAG (Danielskry) | ~2K | None stated | **Yes** | Frameworks |
| e2b-dev/awesome-ai-agents | 11K+ | Via form or PR | **Yes** | Memory/Tools |
| awesome-langchain (kyrolabs) | 7K+ | None stated | **Yes** | Other LLM Frameworks |

---

## TIER 1 — Submit NOW (no star minimum)

### 1. Awesome-Agent-Memory (TeleAI-UAGI)
**Repo:** https://github.com/TeleAI-UAGI/Awesome-Agent-Memory
**Stars:** ~500
**Why:** This is THE list for AI agent memory. Cognee, Mem0, Zep, Letta are already there.
**Section:** Open-Source Products
**Entry format:** Star badge + code/paper/blog links

**Entry to add:**
```markdown
**[Aura](https://github.com/teolex2020/AuraSDK)** ![GitHub stars](https://img.shields.io/github/stars/teolex2020/AuraSDK?style=social) [[code](https://github.com/teolex2020/AuraSDK)] [[docs](https://github.com/teolex2020/AuraSDK/blob/main/docs/API.md)]
```

**PR title:** Add Aura — Rust-native cognitive memory engine for AI agents

**PR body:**
```
Add AuraSDK to the Open-Source products section.

Aura is a cognitive memory engine for AI agents written in pure Rust with Python bindings (PyO3). Key differentiators from existing entries:

- **No LLM/embedding required** — uses SDR + MinHash + Tag Jaccard RRF fusion for recall
- **Sub-millisecond latency** — store 0.09ms, recall 0.62ms (vs 200ms+ for embedding-based)
- **Cognitive lifecycle** — 4-level decay (Working/Decisions/Domain/Identity), 8-phase maintenance
- **Trust & provenance** — source_type tracking (recorded/retrieved/inferred/generated)
- **2.7 MB binary** — works fully offline, no dependencies

PyPI: https://pypi.org/project/aura-memory/
MIT License, 225 Rust tests, 101 Python tests.
```

---

### 2. Awesome-AI-Agents (Jenqyang)
**Repo:** https://github.com/Jenqyang/Awesome-AI-Agents
**Stars:** ~1K
**Why:** Has a "Tools" section where Mem0 is already listed. Direct fit.
**Section:** Applications > Tools

**Entry to add:**
```markdown
- [AuraSDK](https://github.com/teolex2020/AuraSDK) - Rust-native cognitive memory for AI agents with <1ms recall, trust scoring, and 4-level hierarchical decay. No LLM required. ![GitHub Repo stars](https://img.shields.io/github/stars/teolex2020/AuraSDK)
```

**PR title:** Add AuraSDK to Tools section — cognitive memory for AI agents

**PR body:**
```
Adding AuraSDK to the Tools section alongside mem0 and other agent infrastructure.

AuraSDK is a persistent memory SDK for AI agents. Written in Rust with Python bindings.
Differs from mem0 in that it requires no LLM or embedding model — recall uses
multi-signal RRF fusion (SDR + MinHash + Tag Jaccard) with sub-millisecond latency.

Includes trust scoring, namespace isolation, 4-level decay hierarchy, and ChaCha20 encryption.
```

---

### 3. e2b-dev/awesome-ai-agents
**Repo:** https://github.com/e2b-dev/awesome-ai-agents
**Stars:** 11K+
**Submission:** PR or Google Form: https://forms.gle/UXQFCogLYrPFvfoUA

**Entry format (detailed, with collapsible details):**
```markdown
## [AuraSDK](https://github.com/teolex2020/AuraSDK)
Cognitive memory engine for AI agents — <1ms recall, no LLM required

<details>

### Category
Developer Tools, Memory

### Description
- Rust-native memory SDK with Python bindings (PyO3)
- 4-signal RRF fusion recall: SDR + MinHash + Tag Jaccard + optional embeddings
- 4-level hierarchical decay: Working → Decisions → Domain → Identity
- Trust scoring with source provenance (recorded/retrieved/inferred/generated)
- Works fully offline, 2.7 MB binary, ChaCha20 encryption
- Integrations: Ollama, LangChain, OpenAI, CrewAI, AutoGen, MCP Server

### Links
- [GitHub](https://github.com/teolex2020/AuraSDK)
- [PyPI](https://pypi.org/project/aura-memory/)
- [API Docs](https://github.com/teolex2020/AuraSDK/blob/main/docs/API.md)
- [Colab Notebook](https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb)

</details>
```

---

### 4. Awesome-RAG (Danielskry)
**Repo:** https://github.com/Danielskry/Awesome-RAG
**Stars:** ~2K
**Section:** Frameworks that Facilitate RAG
**Why:** Aura can be positioned as a lightweight alternative to full RAG — memory layer without the pipeline overhead.

**Entry to add:**
```markdown
- [AuraSDK](https://github.com/teolex2020/AuraSDK): Rust-native cognitive memory for AI agents with sub-millisecond recall using RRF fusion (no embeddings required). Supports pluggable embeddings as optional 4th signal.
```

**PR title:** Add AuraSDK — lightweight memory framework with RRF fusion recall

**PR body:**
```
Adding AuraSDK to the Frameworks section.

AuraSDK provides a memory/retrieval layer for AI agents using multi-signal
RRF fusion (SDR + MinHash + Tag Jaccard + optional embeddings) instead of
pure vector search. Sub-millisecond recall, works offline, no embedding model
required. Can be used alongside RAG pipelines for agent runtime memory.
```

---

### 5. awesome-langchain (kyrolabs)
**Repo:** https://github.com/kyrolabs/awesome-langchain
**Stars:** 7K+
**Section:** Other LLM Frameworks
**Why:** Aura has a LangChain integration example.

**Entry to add:**
```markdown
- [AuraSDK](https://github.com/teolex2020/AuraSDK) - Rust-native cognitive memory for AI agents. <1ms recall, no LLM needed. [LangChain integration example](https://github.com/teolex2020/AuraSDK/blob/main/examples/langchain_agent.py). ![GitHub Repo stars](https://img.shields.io/github/stars/teolex2020/AuraSDK)
```

---

## TIER 2 — Submit after reaching star milestones

### 6. awesome-rust (rust-unofficial)
**Repo:** https://github.com/rust-unofficial/awesome-rust
**Stars:** 47K
**Requirement:** Stars > 50 (or crates.io downloads > 2000)
**Section:** Applications > Artificial Intelligence > Machine learning
**Sort:** Alphabetical (would go near top, before "autumnai/leaf")

**Entry to add (exact format per CONTRIBUTING.md):**
```markdown
* [teolex2020/AuraSDK](https://github.com/teolex2020/AuraSDK) [[aura](https://crates.io/crates/aura)] - Cognitive memory engine for AI agents with RRF fusion recall, hierarchical decay, and trust scoring [![build badge](https://github.com/teolex2020/AuraSDK/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/teolex2020/AuraSDK/actions)
```

**PR title:** Add AuraSDK — cognitive memory for AI agents (Rust + PyO3)

**PR body:**
```
Adding AuraSDK to the Machine learning section.

AuraSDK is a cognitive memory engine for AI agents:
- 46 Rust modules (~19,500 lines), 225 tests, 0 unsafe blocks
- SDR + MinHash + Tag Jaccard RRF fusion for sub-millisecond recall
- 4-level hierarchical decay, trust scoring, ChaCha20 encryption
- Python bindings via PyO3, published to crates.io and PyPI

Stars: [current count] (meets >50 threshold)
```

**WAIT UNTIL: 50+ stars**

---

### 7. awesome-python (vinta)
**Repo:** https://github.com/vinta/awesome-python
**Stars:** 228K
**Requirement:** 100+ stars for "Hidden Gem" path; repo must be 6+ months old
**Section:** Machine Learning

**Entry to add:**
```markdown
- [aura-memory](https://github.com/teolex2020/AuraSDK) - Cognitive memory SDK for AI agents with sub-millisecond recall, trust scoring, and hierarchical decay.
```

**PR title:** Add aura-memory as Hidden Gem — cognitive memory for AI agents

**PR body:**
```
Submitting under the Hidden Gem category.

aura-memory solves a niche problem: giving AI agents persistent memory
that doesn't require LLM calls or embeddings. It uses a novel RRF fusion
approach (SDR + MinHash + Tag Jaccard) for sub-millisecond recall.

Unique value over existing entries (LangChain, LlamaIndex):
- No LLM or embedding model required for any operation
- Pure Rust core (2.7 MB binary), Python bindings via PyO3
- Cognitive lifecycle: 4-level decay, trust scoring, maintenance
- Works fully offline with zero dependencies

Stars: [current count] | Active commits | MIT License
```

**WAIT UNTIL: 100+ stars, repo 6+ months old**

---

## Submission Order (recommended)

### Phase 1 — Now (0-50 stars)
1. **Awesome-Agent-Memory** — most relevant, memory-specific list
2. **Awesome-AI-Agents (Jenqyang)** — Tools section, Mem0 already there
3. **awesome-ai-sdks (e2b)** — via Google Form, easy submission

### Phase 2 — After 50 stars
4. **awesome-rust** — big audience, Rust community
5. **Awesome-RAG** — framework section
6. **awesome-langchain** — integration angle

### Phase 3 — After 100 stars
7. **awesome-python** — Hidden Gem path, biggest audience

---

## Tips

- Submit one PR per list, never batch
- Be honest about star count if asked
- Respond to reviewer comments within 24h
- If rejected for low stars, bookmark and resubmit at milestone
- Each successful awesome list placement = steady organic traffic forever
