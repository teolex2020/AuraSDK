---
title: How to give your AI agent memory that actually learns — in 5 lines of Python
published: false
description: Add persistent, trust-scored memory to any AI agent. Sub-millisecond recall, no LLM required, no cloud dependency. Works with Ollama, OpenAI, LangChain, or any LLM.
tags: ai, python, machinelearning, opensource
cover_image:
canonical_url:
---

Your AI agent forgets everything between sessions.

Every conversation starts from zero. Every user preference, every decision, every piece of context — gone. You paste old conversations into the system prompt, hit the token limit, and wonder why the agent feels so... stateless.

Most "memory" solutions bolt on a vector database, call an embedding API, and charge you per query. You now have 200ms latency, a cloud dependency, and a monthly bill — for what is essentially a fancy search index.

What if your agent could remember like a human? Important things stick. Trivial things fade. Trusted sources rank higher than random web scrapes. And it all happens in **under 1 millisecond**, locally, with **zero LLM calls**.

That's [Aura](https://github.com/teolex2020/AuraSDK).

## What makes Aura different

| | Aura | Others (Mem0, Zep, Cognee) |
|---|---|---|
| LLM required | **No** | Yes |
| Recall latency | **<1ms** | 200ms+ / LLM-bound |
| Works offline | **Yes** | No |
| Binary size | **~3 MB** | Heavy |
| Cost per op | **$0** | API billing |

Aura is a Rust-native cognitive memory engine with Python bindings. It uses a 4-signal RRF (Reciprocal Rank Fusion) recall system — no embeddings required — and models memory decay, consolidation, and trust scoring inspired by how human memory actually works.

Let's see how it works.

## 1. Install

```bash
pip install aura-memory
```

That's it. No Docker, no API keys, no cloud account. The entire engine ships as a single ~3 MB binary.

## 2. Store & recall — the basics

```python
from aura import Aura, Level

brain = Aura("./agent_memory")

# Store memories at different importance levels
brain.store("User prefers dark mode and Vim keybindings",
            level=Level.Identity, tags=["preference", "ui"])

brain.store("Deploy staging before production, always run tests",
            level=Level.Decisions, tags=["workflow"])

brain.store("Fix login bug - users getting 403 on /api/auth",
            level=Level.Working, tags=["bug", "auth"])

# Recall — returns formatted context ready for LLM injection
context = brain.recall("authentication issues", token_budget=2000)
print(context)
```

Output:

```
=== COGNITIVE CONTEXT ===
[IDENTITY]
  - User prefers dark mode and Vim keybindings [preference, ui]

[DECISIONS]
  - Deploy staging before production, always run tests [workflow]

[WORKING]
  - Fix login bug - users getting 403 on /api/auth [bug, auth]

=== END CONTEXT ===
```

That's it. `store()` → `recall()` → inject into your system prompt. Five lines to give your agent persistent memory.

## 3. Memory levels — not all memories are equal

Aura organizes memory into 4 levels across 2 tiers, modeled after human cognitive architecture:

| Tier | Level | Decay rate | Use case |
|---|---|---|---|
| **Core** | Identity | 0.99/cycle | User preferences, personality |
| **Core** | Domain | 0.95/cycle | Learned facts, domain knowledge |
| **Cognitive** | Decisions | 0.90/cycle | Choices made, action items |
| **Cognitive** | Working | 0.80/cycle | Current tasks, recent messages |

**Core tier** = slow decay (weeks to months). Your agent's "personality" and knowledge base.
**Cognitive tier** = fast decay (hours to days). Ephemeral context that fades naturally.

This means your agent doesn't need explicit "forget" logic. Old tasks decay away. Core knowledge persists. Just like your brain.

```python
# Query only recent, ephemeral memories
recent = brain.recall_cognitive("workflow")

# Query only long-term knowledge
knowledge = brain.recall_core_tier("programming")
```

## 4. Trust scoring — the killer feature

Here's where Aura gets interesting. Not all information sources are equally reliable. A user telling you their name is more trustworthy than a web scrape claiming "Python 4.0 is coming soon."

```python
from aura import TrustConfig

tc = TrustConfig()
tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
brain.set_trust_config(tc)

# Store from different sources
brain.store("Python 3.13 released October 2024",
            tags=["python"], channel="user")

brain.store("Python 4.0 coming soon",
            tags=["python"], channel="web_scrape")

# Trust-weighted recall ranks user-sourced memory higher
results = brain.recall_structured("python release", top_k=5)
for r in results:
    print(f"  score={r['score']:.3f}  trust={r['trust']:.2f}  {r['content']}")
```

Output:

```
  score=0.995  trust=1.00  Python 3.13 released October 2024
  score=0.589  trust=0.50  Python 4.0 coming soon
```

The user-sourced fact scores **0.995**. The web scrape scores **0.589**. Your agent now has built-in epistemological hygiene — it knows *how much* to trust each piece of information.

This is `source_type` provenance in action. Every memory carries metadata about where it came from, and that metadata directly influences recall ranking.

## 5. Plug it into any LLM

Aura is LLM-agnostic. The pattern is always the same:

```python
user_message = "What are my UI preferences?"

# 1. Recall relevant context
context = brain.recall(user_message, token_budget=2000)

# 2. Build system prompt
system_prompt = f"""You are a helpful assistant with memory.

{context}

Use the above context to personalize your responses."""

# 3. Send to your LLM of choice:
# Ollama:     requests.post("http://localhost:11434/api/chat", ...)
# OpenAI:     openai.chat.completions.create(messages=[...])
# LangChain:  ChatPromptTemplate with {context}
# Claude:     anthropic.messages.create(...)
# Any HTTP:   just inject system_prompt
```

No adapters. No framework lock-in. If your LLM takes a string, Aura works with it.

## Bonus: structured recall with scores

When you need more than formatted text — for routing, filtering, or debugging:

```python
results = brain.recall_structured("user preferences", top_k=5)
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} -- {r['content'][:80]}")
```

```
  [IDENTITY] score=0.590 -- User prefers dark mode and Vim keybindings
  [WORKING]  score=0.586 -- Fix login bug - users getting 403 on /api/auth
  [DECISIONS] score=0.581 -- Deploy staging before production, always run tests
```

Each result includes level, score, trust, tags, timestamps, and source metadata. Use this to build intelligent routing: high-trust Identity memories go straight to the system prompt; low-trust Working memories get verified first.

## Performance — sub-millisecond, for real

Benchmarked on a standard machine with 1,000 stored records:

| Operation | Latency |
|---|---|
| Store | 0.129 ms/op |
| Recall (1K records) | 0.861 ms/op |
| Search by tag | 0.103 ms/op |

For comparison, embedding-based recall typically runs **200ms+** per call. Aura is **200x faster** because it uses SDR (Sparse Distributed Representation) encoding + MinHash + tag matching — no neural network inference needed.

You *can* optionally add embeddings as a 4th signal if you want semantic similarity on top:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
brain.set_embedding_fn(lambda text: model.encode(text).tolist())
```

But the 3-signal fusion works great without them.

## Living memory — decay, reflect, consolidate

Run a single maintenance cycle and Aura handles the rest:

```python
report = brain.run_maintenance()
print(f"Decayed: {report.decay.decayed}")       # Strength reduced
print(f"Promoted: {report.reflect.promoted}")     # Important memories promoted
print(f"Consolidated: {report.consolidation.native_merged}")  # Duplicates merged
print(f"Archived: {report.records_archived}")     # Weak memories archived
```

Call this periodically (every N interactions, or on a schedule), and your agent's memory stays clean and relevant without manual curation.

## More features you get out of the box

- **Namespace isolation** — keep test/prod/per-user memories separate
- **Encryption** — ChaCha20-Poly1305 + Argon2id, one argument: `Aura("./data", password="secret")`
- **MCP server** — expose memory as a tool for Claude, GPT, or any MCP-compatible agent
- **Zero dependencies** — pure Rust core, no runtime requirements

## Try it now

The fastest way to try Aura is the interactive Colab notebook — zero setup, runs in your browser:

**[Open in Google Colab](https://colab.research.google.com/github/teolex2020/AuraSDK/blob/main/examples/colab_quickstart.ipynb)**

Or install locally:

```bash
pip install aura-memory
```

- [GitHub](https://github.com/teolex2020/AuraSDK) — star it if you find it useful
- [API docs](https://github.com/teolex2020/AuraSDK/blob/main/docs/API.md) — full reference for 40+ methods
- [Examples](https://github.com/teolex2020/AuraSDK/tree/main/examples) — Ollama integration, LangChain, and more

---

Aura is MIT-licensed and built by a solo developer. If you're building AI agents that need to remember, I'd love to hear what you think.
