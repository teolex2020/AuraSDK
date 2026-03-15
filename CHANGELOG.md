# Changelog

## 1.4.1

This release completes the full 5-layer cognitive recall pipeline and ships a convenience API for enabling it in one call.

### Added

- **Phase 4d — `PolicyRerankMode::Off | Limited`**
  - Policy hints now shape recall ranking as the final bounded signal
  - Pipeline: `Belief (±5%) → Concept (±4%) → Causal (±3%) → Policy (±2%)`
  - `Prefer`/`Recommend` hints boost relevant records; `Avoid` hints slightly downrank
  - All scope guards retained: min 4 results, top_k ≤ 20, coverage > 0
  - `set_policy_rerank_mode()` / `get_policy_rerank_mode()` API

- **`enable_full_cognitive_stack()` / `disable_full_cognitive_stack()`**
  - Single-call convenience API to activate or deactivate all four cognitive reranking phases
  - Available from both Rust and Python

- **Python bindings for all cognitive mode setters**
  - `aura.enable_full_cognitive_stack()`
  - `aura.disable_full_cognitive_stack()`
  - `aura.set_belief_rerank_mode("off" | "shadow" | "limited")`
  - `aura.set_concept_surface_mode("off" | "inspect" | "limited")`
  - `aura.set_causal_rerank_mode("off" | "limited")`
  - `aura.set_policy_rerank_mode("off" | "limited")`

- **A/B quality benchmark** (`tests/quality_benchmark.rs`)
  - Proves All-Limited pipeline is not worse than All-Off across Precision@K, MRR, NDCG@K
  - Ground-truth labeled corpus with known relevant IDs

- **`ConceptSurfaceMode::Off | Inspect | Limited`**
  - `Inspect` exposes bounded surfaced concepts and per-record annotations
  - `Limited` activates concept reranking as Phase 4b in the recall pipeline
  - Runtime concept-surface telemetry in maintenance reporting

### Production-Relevant

- Full cognitive recall pipeline active: Belief → Concept → Causal → Policy (all bounded)
- Policy surfaced output: stable advisory API
- `enable_full_cognitive_stack()` recommended for new integrations

### Advisory / Inspect Only

- Concept surfaced output (`get_surfaced_concepts()`)
- Causal surfaced patterns (`get_surfaced_causal_patterns()`)
- Policy surfaced hints (`get_surfaced_policy_hints()`)

### Safety Guarantees Preserved

- No LLM dependency introduced
- No cloud dependency introduced
- All rerank phases bounded: score cap + positional shift cap + scope guards
- Deterministic: same query always returns same order (cache-hit path)
- Zero result removal: downrank ≠ remove

### Validation

- Full suite green at release: `828 passed, 0 failed`
- Policy Limited eval: 10 tests (no degradation, score bounds, soak)
- Full stack eval: 12 combined-mode tests
- Quality benchmark: 9 A/B tests (MRR, P@K, NDCG@K)
