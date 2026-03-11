# Belief Grouping Densification Sprint — Report

**Date**: 2026-03-11
**Sprint goal**: Remove upstream belief formation bottleneck blocking Candidate C (concept-assisted grouping) on practical corpora.

## Problem Statement

The current belief coarse key `namespace:sorted_tags(top3):semantic_type` fragments corpora with diverse tags into unique belief corridors, producing singleton beliefs with low seed density per concept partition. This prevents concept formation on practical (non-curated) corpora.

**Two-lever bottleneck identified**: Both the coarse key AND the SDR subclustering threshold (CLAIM_SIMILARITY_THRESHOLD=0.15) contribute. Records with related but non-identical content have Tanimoto similarity ~0.10-0.14 — below the 0.15 threshold — so even when the coarse key groups them together, SDR subclustering splits them into individual clusters of size 1.

## Variants Tested

| Config | Coarse Key | SDR Threshold | Description |
|---|---|---|---|
| **Standard** (baseline) | `ns:sorted_tags(top3):st` | 0.15 | Current production key |
| **TopOneTag** | `ns:top_1_tag:st` | 0.15 | Reduces tag fragmentation |
| **SemanticOnly** | `ns:st` | 0.15 | No tag component at all |
| **Semantic+0.10** | `ns:st` | 0.10 | SemanticOnly + lowered threshold |

## Implementation

- `CoarseKeyMode` enum added to `BeliefEngine` (Standard, TopOneTag, SemanticOnly)
- `claim_similarity_override: Option<f32>` field for per-engine threshold override
- `claim_key_with_mode()` dispatches key generation per mode
- `sdr_subcluster()` accepts explicit threshold parameter
- Aura API: `set_belief_coarse_key_mode()`, `set_belief_similarity_threshold()`
- All changes backward-compatible: Standard mode + default threshold = unchanged behavior

## Results

### Quality Benchmark (curated 26-record, 13-cluster dataset)

| Config | Precision | Recall | F1 | False Merge Rate |
|---|---|---|---|---|
| **Standard** | 1.000 | 0.824 | 0.903 | 0.000 |
| **TopOneTag** | 1.000 | 0.824 | 0.903 | 0.000 |
| **SemanticOnly** | 0.778 | 0.824 | 0.800 | 0.013 |
| **Semantic+0.10** | 0.233 | 0.824 | 0.364 | 0.149 |

**Analysis**: TopOneTag preserves full quality. SemanticOnly introduces mild false merges (1.3%). Semantic+0.10 causes catastrophic false merge rate (14.9%) — precision collapses from 1.000 to 0.233.

### Combined Practical Density (22 records, 8 cycles)

| Config | Beliefs | Seeds | Partitions ≥2 seeds | Concepts |
|---|---|---|---|---|
| **Standard** | 3 | 3 | 1 | 0 |
| **TopOneTag** | 5 | 4→5 | 1 | 0 |
| **SemanticOnly** | 7 | 2 | 0 | 0 |
| **Semantic+0.10** | 9 | 2 | 0 | 0 |

**Analysis**: Loosening the coarse key increases belief count but does NOT reliably increase seed density per concept partition. No variant produces concepts. The bottleneck is not just fragmentation — it's that practical corpora have too few same-topic records within each concept partition to pass the MIN_BELIEF_STABILITY and MIN_BELIEF_CONFIDENCE gates.

### Concept Practical Coverage (4 scenario corpora)

| Config | Concepts Formed | Avg Coverage |
|---|---|---|
| **Standard** | 0 | 0.0% |
| **TopOneTag** | 0 | 0.0% |
| **SemanticOnly** | 0 | 0.0% |
| **Semantic+0.10** | 0 | 0.0% |

**Analysis**: Zero concept formation across all variants and all 4 practical corpora (deploy-chain, stable-preference, multi-topic, contextual). Densification alone does not unblock Candidate C.

### Candidate B Safety

All 4 configs pass: belief reranking is unaffected by coarse key changes.

### Cross-Layer Stack

All 4 configs pass: concept/causal/policy layers function normally.

### Churn Stability

All 4 configs pass: max churn over last 5 of 10 cycles = 0.000.

## Diagnostic Deep-Dive

### Deploy-Chain Corpus (6 records)
- Standard: `key = default:deploy,staging:decision` — 0 beliefs (only 3 records share exact tag set, Tanimoto ~0.13 < 0.15)
- SemanticOnly: `key = default:decision` — 0 beliefs (all 6 in same corridor, but Tanimoto < 0.15 splits them all)
- Semantic+0.10: 1 belief formed (Tanimoto ~0.13 > 0.10 threshold merges deploy records)
  - state=Singleton, stability=5.0, confidence=0.900

### Tanimoto Distribution (deploy corpus)
```
Tanimoto(0, 1): 0.1288  (same-topic deploy records)
Tanimoto(0, 2): 0.1294  (same-topic deploy records)
Tanimoto(1, 2): 0.1446  (same-topic deploy records)
Tanimoto(0, 3): 0.0351  (cross-topic: deploy vs safety)
Tanimoto(1, 3): 0.0119  (cross-topic: deploy vs safety)
```

**Key insight**: Same-topic practical records have Tanimoto 0.10-0.15. The 0.15 threshold sits RIGHT at the boundary — too high to group them. Lowering to 0.10 groups same-topic records but also introduces false merges on the curated benchmark (14.9%).

## Verdict

### `NO SAFE DENSIFICATION`

**Rationale**:
1. **TopOneTag** (safe, precision=1.000) produces no additional concept density — it only helps when tag diversity is the sole bottleneck, which it isn't here
2. **SemanticOnly** (marginal, precision=0.778) produces more beliefs but no additional concept seeds — quality degradation without benefit
3. **Semantic+0.10** (unsafe, precision=0.233) produces beliefs from practical records but at catastrophic false merge cost — unacceptable
4. **No variant produces concepts** — the concept formation bottleneck is deeper than coarse key fragmentation

### Root Cause Analysis

The concept formation bottleneck has THREE layers, not two:
1. **Coarse key fragmentation** — solved by SemanticOnly, but insufficient alone
2. **SDR subclustering threshold** — 0.15 is too high for practical paraphrases (Tanimoto ~0.12-0.14), but lowering to 0.10 causes false merges
3. **Corpus size** — practical test corpora have 3-6 records per topic, insufficient to form 2+ beliefs per concept partition even when grouping succeeds

### Recommendations

1. **Do NOT deploy any coarse key variant** — no safe density gain found
2. **Keep TopOneTag/SemanticOnly infrastructure** — useful for future experiments
3. **Candidate C remains SHADOW OBSERVE** — concept formation on practical corpora needs a different approach (e.g., larger test corpora, belief-aware seeding, or relaxed concept seed gates)
4. **Concept cross-topic merge fix**: Added tag barrier to `cluster_beliefs()` — beliefs must share ≥1 tag to merge. Prevents false merges from incidental n-gram overlap at threshold 0.10

## Files Changed

- **src/belief.rs**: `CoarseKeyMode` enum, `claim_key_with_mode()`, `claim_similarity_override`, parameterized `sdr_subcluster()`
- **src/aura.rs**: `set_belief_coarse_key_mode()`, `set_belief_similarity_threshold()` API surface
- **src/concept.rs**: Tag barrier in `cluster_beliefs()` — prevents cross-topic false merges at threshold 0.10
- **tests/belief_densification.rs**: 9 tests across 4 configurations (NEW)
- **BELIEF_DENSIFICATION_REPORT.md**: this report (NEW)

## Test Summary

| Test | Status |
|---|---|
| `densification_mode_diagnostic` | PASS |
| `densification_partition_density_per_variant` | PASS |
| `densification_quality_benchmark_per_variant` | PASS |
| `densification_concept_practical_coverage` | PASS |
| `densification_candidate_b_not_regressed` | PASS |
| `densification_cross_layer_stack_intact` | PASS |
| `densification_churn_stability` | PASS |
| `densification_combined_practical_density` | PASS |
| `densification_report_emits_variant_comparison` | PASS |

Full test suite: 385 lib + all integration tests PASS. Zero failures.
