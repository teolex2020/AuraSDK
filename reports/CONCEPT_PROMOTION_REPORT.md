# Concept Inspection-Only Promotion Sprint — Report

**Date**: 2026-03-11
**Sprint goal**: Promote Candidate C from "safe redesign found" to inspection-only public feature.

## Background

The Concept Representation Redesign Sprint (preceding this sprint) unblocked concept formation:
- Fixed `parse_belief_key_ns_st` partition bug (primary enabler)
- Added canonical feature tokenization + Jaccard similarity (Variant A)
- Canon+TF+Rlx: 3 concepts, 78–88% coverage, 3/3 topics, 0 false merges
- Full suite green (542 tests)

This sprint adds the public inspection surface: bounded, sorted, provenance-complete concept output that does NOT affect recall, compression, or behavior.

## Implementation

### SurfacedConcept Type (`src/concept.rs`)

Stable external contract, decoupled from internal `ConceptCandidate`:

```rust
pub struct SurfacedConcept {
    pub id: String,
    pub key: String,
    pub state: String,           // "stable" or "candidate"
    pub namespace: String,
    pub semantic_type: String,
    pub core_terms: Vec<String>,
    pub shell_terms: Vec<String>,
    pub tags: Vec<String>,
    pub abstraction_score: f32,
    pub confidence: f32,
    pub cluster_size: usize,
    pub support_mass: f32,
    pub belief_ids: Vec<String>,
    pub record_ids: Vec<String>,
}
```

### Surface Rules

| Rule | Value |
|------|-------|
| MAX_SURFACED_CONCEPTS | 10 |
| MAX_SURFACED_PER_NAMESPACE | 5 |
| Stable concepts | Always eligible |
| Candidate threshold | abstraction_score >= 0.70 |
| Rejected concepts | Never surfaced |
| Provenance required | belief_ids + record_ids non-empty |
| Content required | core_terms or tags non-empty |

### Sorting (deterministic)

1. Higher abstraction_score
2. Higher confidence
3. Larger cluster_size
4. Stable over Candidate
5. Key tiebreak (alphabetical)

### APIs (`src/aura.rs`)

- `get_surfaced_concepts(limit: Option<usize>) -> Vec<SurfacedConcept>`
- `get_surfaced_concepts_for_namespace(namespace: &str, limit: Option<usize>) -> Vec<SurfacedConcept>`

Both are inspection-only — they read concept state but do not mutate anything.

## Results

### Realistic Corpus (33 records, 3 topics)

Configuration: Canon+TF+Rlx (best known config)

| Concept | State | Score | Conf | Cluster | Tags | Core Terms |
|---------|-------|-------|------|---------|------|------------|
| database:decision | stable | 0.845 | 0.900 | 5 | [database] | [database, query] |
| editor:preference | stable | 0.837 | 0.900 | 6 | [editor] | [dark, theme] |
| deploy:decision | candidate | 0.746 | 0.900 | 6 | [deploy] | [] |

- 3 surfaced concepts from 3 topics
- 2 stable, 1 strong candidate (score 0.746 >= 0.70 threshold)
- Full provenance: every concept traces to specific beliefs and records
- Core terms capture the "essence" of each concept cluster

### Safety Gates

| Gate | Result | Details |
|------|--------|---------|
| No recall mutation | **PASS** | Identical recall results regardless of concept similarity mode |
| No record mutation | **PASS** | Surfaced concepts are read-only snapshots |
| No hidden control path | **PASS** | Surface functions only read ConceptEngine state |
| No cross-topic bleed | **PASS** | All records in each concept share common topic tags |
| No provenance gaps | **PASS** | Every concept has valid belief_ids and record_ids; all references resolve |
| Deterministic sorting | **PASS** | Identical output on repeated calls |
| Bounded output | **PASS** | Respects MAX_SURFACED_CONCEPTS=10, MAX_SURFACED_PER_NAMESPACE=5, explicit limit |
| Namespace filter | **PASS** | Correctly filters by namespace, returns empty for nonexistent |
| Replay stability | **PASS** | Max stable streak >= 3 consecutive cycles (measured: 5) |
| Weak/rejected filtered | **PASS** | Only stable + strong candidates surfaced |

### Provenance Validation

For each surfaced concept:
- All `belief_ids` resolve to actual beliefs in the engine
- All `record_ids` resolve to actual records in storage
- `cluster_size == belief_ids.len()` (consistent)
- Records within each concept share common topic tags (no cross-topic merge)

## Acceptance Criteria

| Criterion | Result |
|-----------|--------|
| Surfaced concepts non-empty on realistic corpus | **PASS** — 3 concepts from 3 topics |
| Grouping useful and bounded | **PASS** — core_terms capture essence, bounded by limits |
| No false merges | **PASS** — 0 cross-topic merges |
| No recall impact | **PASS** — identical recall regardless of concept mode |
| Deterministic across replay | **PASS** — stable key identity for 5+ consecutive cycles |
| Provenance complete | **PASS** — all belief_ids and record_ids resolve |

## Verdict

### `CANDIDATE C: PROMOTED (INSPECTION-ONLY)`

Surfaced concept output is:
- **Useful**: meaningful grouping with core_terms, tags, and provenance
- **Safe**: zero impact on recall, compression, or behavior
- **Bounded**: respects global and per-namespace limits
- **Deterministic**: stable across calls and replay cycles
- **Provenance-complete**: full chain from concept → beliefs → records

### What C IS now
- Inspection-only concept grouping surface
- Public API for concept discovery and analysis
- Read-only access to cognitive concept state

### What C is NOT
- NOT recall influence (concepts don't affect ranking)
- NOT compression (concepts don't merge or reduce records)
- NOT behavioral cognition (concepts don't drive policy)
- NOT hidden control logic (all surfacing is explicit)

## Updated Candidate Status

| Candidate | Status | Mode |
|-----------|--------|------|
| A (surfaced policy) | STABLE | Production API |
| B (belief rerank) | FULLY HARDENED | Limited mode with guardrails |
| C (concept grouping) | **PROMOTED (inspection-only)** | Surfaced concepts API |

## Files Changed

- **src/concept.rs**: `SurfacedConcept` struct, `surface_concepts()`, `surface_concepts_filtered()`, constants
- **src/aura.rs**: `get_surfaced_concepts()`, `get_surfaced_concepts_for_namespace()`
- **tests/concept_promotion.rs**: 12 tests (6 unit + 6 integration) (NEW)
- **CONCEPT_PROMOTION_REPORT.md**: this report (NEW)

## Test Summary

| Test | Type | Status |
|------|------|--------|
| `stable_concepts_are_surfaced` | unit | PASS |
| `strong_candidates_can_be_surfaced` | unit | PASS |
| `weak_or_rejected_concepts_are_filtered` | unit | PASS |
| `surfaced_concepts_require_provenance` | unit | PASS |
| `surface_sorting_is_deterministic` | unit | PASS |
| `surface_limit_is_enforced` | unit | PASS |
| `surfaced_concepts_non_empty_on_realistic_corpus` | integration | PASS |
| `surfaced_concepts_zero_recall_impact` | integration | PASS |
| `surfaced_concepts_respect_namespace_filter` | integration | PASS |
| `surfaced_concepts_have_full_provenance` | integration | PASS |
| `surfaced_concepts_do_not_cross_topic_merge` | integration | PASS |
| `surfaced_concepts_stable_across_replay` | integration | PASS |

All 12 tests PASS. Full suite (554 tests) green.
