# Changelog

## Unreleased / Aura v2 Cognitive Upgrade

This release turns Aura from a strong local memory engine into a bounded cognitive stack with production-safe promotion of the first higher-order layers.

### Added

- Belief layer with SDR-based claim grouping, competing hypotheses, resolved/unresolved state, churn metrics, and persistence.
- Concept layer with canonical-feature similarity, surfaced inspection output, deterministic identity, and zero recall impact.
- Causal pattern layer for advisory-only temporal and explicit causal discovery.
- Policy hint layer with surfaced inspection output, deterministic templates, provenance, and bounded filtering.
- Maintenance observability:
  - per-phase timings
  - per-layer stability metrics
  - inspection helpers for beliefs, concepts, causal patterns, and policy hints
- Surfaced APIs:
  - `get_surfaced_policy_hints()`
  - `get_surfaced_policy_hints_for_namespace()`
  - `get_surfaced_concepts()`
  - `get_surfaced_concepts_for_namespace()`
- Belief-aware recall reranking with bounded influence:
  - tri-state mode `Off | Shadow | Limited`
  - bounded score effect
  - positional shift cap
  - scope guards

### Changed

- Record epistemic model expanded with:
  - `confidence`
  - `support_mass`
  - `conflict_mass`
  - `volatility`
- Concept similarity no longer depends only on raw SDR n-gram similarity for practical corpora.
- Concept parsing fixed so subclustered belief keys do not masquerade as distinct semantic types.
- Diagnostics now distinguish raw recall, shadow scoring, and limited rerank paths cleanly.

### Hardened

- Candidate A: `policy surfaced output` is stable.
- Candidate B: `belief-aware recall` is fully hardened after replay, soak, shadow, wider-rollout, and 100-run synthetic campaign validation.
- Candidate C: `concept grouping` is promoted in inspection-only mode after safe representation redesign and realistic corpus validation.

### Safety Guarantees Preserved

- No LLM dependency introduced.
- No cloud dependency introduced.
- No recall mutation from concept/policy surfaces.
- No record mutation from promoted higher layers.
- Deterministic sorting and bounded outputs on surfaced APIs.

### Validation

- Full suite green at latest recorded milestone: `554 tests, 0 failures`.

