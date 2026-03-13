# Changelog

## 1.4.1

This release keeps Aura honest as a deterministic local memory engine and ships the newer cognitive work in bounded, advisory form.

### Added

- `ConceptSurfaceMode::Off | Inspect | Limited`
  - `Inspect` exposes bounded surfaced concepts
  - bounded per-record concept annotations are available on inspect-mode recall surfaces
  - `Limited` remains intentionally blocked
- Runtime concept-surface telemetry in maintenance reporting:
  - surfaced concept count
  - covered namespace count
  - global / namespace / record annotation call counters
- Dedicated internal rollout artifacts for concept promotion review and inspect-mode trialing

### Production-Relevant

- Belief-aware recall reranking remains the only promoted cognitive influence path
- Policy surfaced output remains advisory output only

### Advisory / Inspect Only

- Concept surfaced output
- Per-record concept annotations
- Causal surfaced patterns
- Policy hints beyond surfaced reporting

### Safety Guarantees Preserved

- No LLM dependency introduced
- No cloud dependency introduced
- No concept influence on recall ordering
- No causal or policy behavior influence
- Deterministic bounded surfaced APIs retained

### Validation

- Full suite green at latest release-prep checkpoint: `414 passed, 0 failed`
