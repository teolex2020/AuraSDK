# AuraSDK Roadmap

> Post-launch evolution plan. Prioritized by impact/effort ratio.
> Updated: 2026-03-03

---

## Phase 1 — Community & Trust (Week 1-2)

Foundation for open-source growth. Unblocks contributors and builds credibility.

### 1.1 CONTRIBUTING.md + Issue Templates
- [x] `CONTRIBUTING.md` — dev setup, PR guidelines, code style, test requirements
- [x] `.github/ISSUE_TEMPLATE/bug_report.md`
- [x] `.github/ISSUE_TEMPLATE/feature_request.md`
- [x] `.github/PULL_REQUEST_TEMPLATE.md`
- **Why:** No community infra = no contributors. This is a gate.

### 1.2 Automated Benchmarking Suite
- [x] `benchmarks/bench_store.py` — 1K / 10K / 100K records
- [x] `benchmarks/bench_recall.py` — cold / warm / cached latency
- [x] `benchmarks/bench_maintenance.py` — full cycle at scale
- [ ] `benchmarks/bench_vs_mem0.py` — head-to-head comparison (optional dep)
- [x] `benchmarks/README.md` — how to reproduce
- [ ] GitHub Action: run benchmarks on release, publish results
- **Why:** README claims <1ms recall and ~270x faster than Mem0. Reproducible proof converts skeptics.

### 1.3 Public Roadmap in README
- [x] Add "Roadmap" section to README.md linking to this file
- **Why:** Shows the project is alive and has direction.

---

## Phase 2 — Ecosystem Gaps (Week 2-4)

Close the biggest holes in framework coverage.

### 2.1 LlamaIndex Integration
- [x] `examples/llamaindex_agent.py` — custom `ChatMemoryBuffer` or `BaseChatStore`
- [x] Test with LlamaIndex `ChatEngine` and `QueryEngine`
- [x] Add to README integrations table
- **Why:** Second most popular RAG framework. Missing it is a visible gap.

### 2.2 Temporal Queries API ("What did the agent know on date X?")
- [x] Rust: `recall_at(query, timestamp)` — filter records by created_at <= timestamp
- [x] Rust: `history(record_id)` — access/strength timeline for a record
- [x] Python: expose both methods via PyO3
- [x] Tests: store records across simulated dates, recall at specific points (11 tests)
- **Why:** Zep's killer feature. Closing this eliminates their main advantage.

### 2.3 Event Callbacks (on_store / on_recall)
- [x] Python: `brain.on_store(callback, tags=None, level=None)` — fire on matching stores
- [x] Python: `brain.on_recall(callback)` — fire after recall with results
- [x] Python: `brain.on_maintenance(callback)` — fire after maintenance cycle
- [x] Allow multiple listeners, unsubscribe by handle (18 tests)
- **Why:** Production agents need reactive patterns, not just request/response.

---

## Phase 3 — Drop-in Adoption (Week 4-6)

Make Aura a zero-friction replacement in existing stacks.

### 3.1 LangChain Memory Class (pip-installable)
- [x] `aura.langchain` module (built into aura-memory, no separate package needed)
- [x] `AuraChatMessageHistory(BaseChatMessageHistory)` — full interface (13 tests)
- [x] `AuraMemory` — duck-typed compatible, works as drop-in replacement
- [x] Works as drop-in replacement: `memory=AuraMemory(brain)`
- [ ] PyPI publish (included in next aura-memory release)
- **Why:** Mem0 has `mem0-langchain`. Developers expect `pip install` and go.

### 3.2 FastAPI Middleware Template
- [x] `examples/fastapi_middleware.py` — inject Aura context into request lifecycle
- [x] Per-user memory isolation via namespace (from auth header)
- [x] Automatic store of conversation turns
- **Why:** FastAPI is the default backend for AI apps. A template saves hours.

### 3.3 Anthropic Claude SDK Integration Example
- [x] `examples/claude_sdk_agent.py` — using `anthropic` Python SDK with Aura memory
- [x] System prompt injection pattern with `brain.recall()`
- [x] Tool-use pattern: Claude calls store/recall as tools
- **Why:** Natural pairing given MCP server already exists. Shows the full Anthropic stack.

---

## Phase 4 — New Markets (Week 6-10)

Expand beyond Python.

### 4.1 C FFI Layer + Multi-Language Examples
- [x] `src/ffi.rs` — C-compatible `extern "C"` API (open, store, recall, recall_structured, maintenance, count)
- [x] `examples/aura.h` — C header file for all languages
- [x] `examples/go/main.go` — CGo FFI binding demo
- [x] `examples/csharp/Program.cs` — P/Invoke demo (.NET 8+)
- [ ] `examples/java/` — JNI or Panama FFI demo
- **Why:** Proves "Pure Rust Core" claim. Shows Aura is a platform, not just a Python library.

### 4.2 TypeScript/WASM Bindings
- [x] Abstract storage layer: `StorageBackend` trait with `FsBackend` + `MemoryBackend` (7 tests)
- [ ] `wasm` feature flag that excludes filesystem-dependent modules
- [ ] `aura-js/` — WASM build via `wasm-pack`
- [ ] NPM package: `@aura-memory/core`
- [ ] `examples/vercel-ai-sdk/` — integration with Vercel AI SDK
- **Why:** JS/TS is ~50% of AI app development. Storage abstraction done, WASM build next.
- **Status:** Storage abstraction complete (`src/backend.rs`). Gradual module migration in progress.

### 4.3 Cloudflare Workers / Edge Runtime
- [ ] WASM build optimized for edge (<1 MB)
- [ ] `examples/cloudflare-worker/` — memory-augmented AI endpoint
- [ ] Use `lite` feature flag (16K-bit SDR)
- **Why:** Edge AI is growing fast. Depends on 4.2 WASM completion.

---

## Phase 5 — Enterprise Readiness (Week 10-14)

Features that unlock paid/enterprise adoption.

### 5.1 Prometheus / OpenTelemetry Metrics
- [x] `/metrics` endpoint in `server` feature (already existed, enhanced)
- [x] Metrics: store_duration, recall_duration, record_count, record_count_by_level, plasticity stats, delete/batch counters
- [x] OpenTelemetry spans: `telemetry` feature flag with OTLP export (`src/telemetry.rs`)
- [x] `#[instrument]` spans on 17 key functions: store, recall_pipeline, recall_core, recall_full, recall_structured, decay, consolidate, reflect, run_maintenance, feedback, supersede, snapshot, rollback, export/import_context, storage append/flush
- [x] Grafana dashboard template (`examples/grafana_dashboard.json`)
- **Why:** Enterprise teams won't adopt without observability.

### 5.2 ~~Distributed Sync (CRDT)~~ — Deferred (no demand)
- Covered by `export_context()` / `import_context()` (Phase 6.3) for manual sharing
- Full CRDT sync would require breaking Record struct (vector clocks) — not justified without user demand
- `crdts` dependency remains available if needed in the future

### 5.3 Multimodal Memory Stubs
- [x] `store_image(path, description)` — stores description with image metadata + auto-tag (9 tests)
- [x] `store_audio_transcript(text, source_path)` — stores transcript with audio provenance (9 tests)
- [x] API-ready, not full multimodal processing — records participate in recall (3 integration tests)
- **Why:** Shows roadmap direction. Competitors claim multimodal via embeddings; we can match with stubs.

### 5.4 Stress Testing & Scale Validation
- [x] `benchmarks/stress_100k.py` — 100K records store + recall + RSS profiling
- [x] `benchmarks/stress_1m.py` — 1M records (memory footprint, recall degradation)
- [ ] Publish results: "Aura handles 1M records with <5ms recall"
- [x] Memory profiling: RSS at 10K / 50K / 100K / 250K / 500K / 1M checkpoints
- **Why:** "Works at scale" needs proof, not claims.

---

## Phase 6 — Competitive Moat (Week 14+)

Long-term features that are hard to replicate.

### 6.1 Adaptive Recall (Learning Query Patterns)
- [x] `brain.feedback(record_id, useful=True/False)` — boost/weaken records based on usefulness
- [x] Auto-boost records that get positive feedback (+0.1 strength)
- [x] Auto-demote records that get negative feedback (-0.15 strength)
- [x] `brain.feedback_stats(record_id)` — returns (positive, negative, net_score) (12 tests)
- **Why:** No competitor has this. Memory that learns what's useful vs noise.

### 6.2 Memory Snapshots & Rollback
- [x] `brain.snapshot(label)` — save current state as JSON
- [x] `brain.rollback(label)` — restore to snapshot (full re-index)
- [x] `brain.diff(label_a, label_b)` — compare two states (added/removed/modified)
- [x] `brain.list_snapshots()` — list available snapshots (13 tests)
- **Why:** Debugging agents requires time-travel. "What changed in memory between run 5 and run 6?"

### 6.3 Agent-to-Agent Memory Sharing Protocol
- [x] `brain.export_context(query, top_k)` — portable JSON fragment with provenance
- [x] `brain.import_context(fragment)` — merge with trust=external, strength halved, tagged "shared"
- [x] Protocol envelope: `{"version": "1.0", "format": "aura_context", "records": [...]}` (7 tests)
- **Why:** Multi-agent systems need shared memory. This becomes a protocol, not just a feature.

### 6.4 Semantic Versioning for Memories
- [x] `brain.supersede(old_id, new_content)` — mark old as superseded, create new with causal link
- [x] `brain.superseded_by(record_id)` — check if superseded, returns new ID
- [x] `brain.version_chain(record_id)` — full version history oldest-to-newest
- [x] Recall prefers latest version (old record strength halved) (15 tests)
- **Why:** Real memory evolves. "User prefers dark mode" from 2024 might be outdated in 2025.

---

## Not Planned (and why)

| Feature | Reason |
|---------|--------|
| GraphQL API | REST + MCP sufficient. Adds complexity without clear demand. |
| Native desktop app | Dashboard covers this. Electron app = huge effort, small ROI. |
| Kubernetes Helm charts | Premature. No enterprise customers yet to justify. |
| Mobile SDK (iOS/Android) | Not our niche. Edge device support covers IoT. |
| Full multimodal processing | Out of scope. Embedding stubs + pluggable fn is enough. |

---

## How to Use This Roadmap

1. Each phase builds on the previous one
2. Items within a phase can be parallelized
3. Check off items as completed: `- [ ]` -> `- [x]`
4. Re-prioritize after each phase based on community feedback
5. Timeframes are estimates for a solo developer — adjust as needed

---

*This roadmap is a living document. Community input welcome via [GitHub Issues](https://github.com/teolex2020/AuraSDK/issues).*
