# Concept Realistic Eval Sprint — Report

**Date**: 2026-03-11
**Sprint goal**: Determine whether realistic corpora (10-15 records/topic) and/or relaxed seed gates unblock concept formation (Candidate C).

## Problem Statement

Previous sprints (densification, representation redesign) found that practical corpora are too small (3-6 records/topic) for concept formation. This sprint tests whether increasing corpus size to 10-15 records/topic, combined with relaxed concept seed gates, can produce non-zero concept coverage without false merges.

## Approach

### Path 1: Realistic Corpora with Standard Gates
- 3 topic corpora: deploy (12 records), database (11 records), editor (10 records)
- Realistic tag diversity: each record has the topic tag + 1-2 sub-tags
- Mixed semantic_types: decision, fact, preference
- Standard concept gates: stability ≥ 2.0, confidence ≥ 0.55

### Path 2: Relaxed Seed Gates (ConceptSeedMode::Relaxed)
- stability ≥ 1.0, confidence ≥ 0.40
- Opt-in via `set_concept_seed_mode(ConceptSeedMode::Relaxed)`
- Same corpora as Path 1

### Additional Tests
- **TagFamily + Relaxed**: Combined best grouping variant (from redesign sprint) with relaxed seed gates
- **4-way comparison**: Standard/TagFamily × Standard/Relaxed on realistic corpus
- **Dense corpus positive control**: 23 records with uniform tags (all same 2 tags + same semantic_type)
- **Dense corpus 4-way**: All 4 configs on dense corpus

## Implementation

- Added `ConceptSeedMode` enum (Standard, Relaxed) to `src/concept.rs`
- Added `seed_mode` field to ConceptEngine with `with_seed_mode()` constructor
- Updated `select_seeds()` to use mode-dependent thresholds
- Added `set_concept_seed_mode()` / `get_concept_seed_mode()` API to `src/aura.rs`
- Added tag barrier to `cluster_beliefs()` — requires shared tags ≥ 1
- Tests: `tests/concept_realistic_eval.rs` (12 tests)

## Results

### Realistic Corpus: 4-Way Comparison

| Config | Beliefs | Seeds | P≥2 | Concepts | Coverage | FM |
|--------|---------|-------|-----|----------|----------|-----|
| Std+Std | 6 | 6 | 0 | 0 | 0.0% | 0 |
| Std+Rlx | 6 | 6 | 0 | 0 | 0.0% | 0 |
| TF+Std | 17 | 17 | 5 | 0 | 0.0% | 0 |
| TF+Rlx | 16 | 16 | 5 | 0 | 0.0% | 0 |

**Analysis**:
- Standard coarse key: 33 records → only 6 beliefs. Deploy topic produces 0 beliefs (every record has unique tag pair → unique coarse key → groups of 1 → skipped).
- TagFamily mode dramatically improves density (6 → 16-17 beliefs, 0 → 5 partitions with ≥2 seeds), but still 0 concepts.
- Relaxed seed gates have no effect: beliefs are Singletons with stability 1-8 and confidence 0.9 — they already pass both Standard and Relaxed gates.

### Dense Corpus: Positive Control

| Config | Beliefs | Seeds | P≥2 | Concepts | Coverage | FM |
|--------|---------|-------|-----|----------|----------|-----|
| Std+Std | 17 | 17 | 8 | 0 | 0.0% | 0 |
| Std+Rlx | 16 | 16 | 7 | 0 | 0.0% | 0 |
| TF+Std | 17 | 17 | 8 | 0 | 0.0% | 0 |
| TF+Rlx | 17 | 17 | 8 | 0 | 0.0% | 0 |

**Analysis**:
- Dense corpus (uniform tags, same semantic_type) produces plenty of beliefs and multiple partitions with ≥2 seeds — but still **0 concepts**.
- This proves the bottleneck is NOT corpus size, NOT tag diversity, NOT seed gates.

### Dense Corpus: Centroid Diagnostics

From the positive control's `ConceptPhaseReport`:

| Cycle | Seeds | Centroids | P≥2 | Pairwise | Above 0.10 | Tanimoto Range |
|-------|-------|-----------|-----|----------|------------|----------------|
| 1 | 6 | 6 | 1 | 1 | 0 | 0.048 |
| 3 | 10 | 10 | 3 | 3 | 0 | 0.048–0.069 |
| 7 | 14 | 14 | 6 | 6 | **0** | 0.048–0.069 |

**Root cause revealed**: Centroid pairwise Tanimoto is **0.048–0.069** — ALL below `CONCEPT_SIMILARITY_THRESHOLD` (0.10). Zero pairs pass the clustering gate. Same-topic belief centroids have only ~5-7% SDR overlap.

### Per-Topic Detail (Relaxed mode)

| Topic | Records | Beliefs | Concepts |
|-------|---------|---------|----------|
| deploy | 12 | 0 | 0 |
| database | 11 | 3 | 0 |
| editor | 10 | 3 | 0 |

Deploy produces 0 beliefs because all 12 records have unique tag pairs → unique coarse keys → groups of 1 → skipped. Database and editor each produce 3 beliefs from records that share identical tag pairs (e.g., `[database, performance]`).

### Safety Gates

| Gate | Result |
|------|--------|
| Cross-topic false merges | 0 across all configs |
| Candidate B safety | PASS |
| Cross-layer stack | PASS |
| Identity stability | N/A (no concepts formed) |

## Root Cause Analysis

The concept formation pipeline has a **3-level bottleneck chain**:

### Level 1: Belief Grouping (coarse key fragmentation)
- Standard coarse key `ns:tags(3):st` creates unique keys for records with different tag combinations
- Deploy records: 12 records → 12 unique coarse keys → 12 groups of 1 → 0 beliefs
- TagFamily mode helps: `ns:first_tag:st` groups by alphabetically first tag → more records per group
- But even TagFamily fragments when sub-tags sort before the topic tag (e.g., "artifacts" < "deploy")

### Level 2: SDR Subclustering (low inter-record Tanimoto)
- Within coarse groups, SDR subclustering at threshold 0.15 further fragments
- Same-topic records use different vocabulary → SDR Tanimoto 0.05-0.14
- Most pairs fall below 0.15 → separate subclusters → separate Singleton beliefs
- Dense corpus: 23 records → 14 Singleton beliefs (each with 1 record)

### Level 3: Concept Clustering (low inter-centroid Tanimoto)
- Singleton belief centroids = single record SDRs
- Inter-centroid Tanimoto: 0.048-0.069 (same topic!)
- `CONCEPT_SIMILARITY_THRESHOLD` = 0.10
- Zero pairs above threshold → 0 clusters with ≥ 2 beliefs → 0 concepts

### Why Relaxed Seed Gates Don't Help
- Beliefs already have confidence 0.900 and stability 1-8
- All beliefs already pass even Standard gates (stability ≥ 2.0, confidence ≥ 0.55 for long-lived ones)
- The bottleneck is clustering, not seed selection

### Why More Records Don't Help
- More records → more unique coarse keys → more groups of 1 → same 0 beliefs (for Standard mode)
- With TagFamily: more records → more beliefs, but SDR subclustering splits them into Singletons
- Singletons have low inter-centroid Tanimoto → still 0 concepts

## The Fundamental Limit

The SDR n-gram approach produces Tanimoto coefficients that reflect **lexical** similarity, not **semantic** similarity. Same-topic records about deployment written with different words ("promoted to canary", "blue-green strategy", "feature flags control rollout") have high semantic similarity but low lexical overlap → low SDR Tanimoto.

For concept formation to work via SDR centroids, records would need to:
- Use very similar vocabulary (near-paraphrases), OR
- Have a semantic embedding layer instead of n-gram SDR

The current architecture is structurally limited: **concepts can only form over beliefs whose records are near-paraphrases** (Tanimoto ≥ 0.15 for belief grouping, then inter-belief centroid Tanimoto ≥ 0.10 for concept clustering). Records about the same topic but with different vocabulary will never cluster.

## Verdict

### `CANDIDATE C: STRUCTURALLY BLOCKED`

No combination of:
- Corpus size (10-15 records/topic tested)
- Coarse key mode (Standard, TagFamily tested)
- Seed gates (Standard, Relaxed tested)
- Uniform tags (positive control tested)

produces non-zero concept coverage. The bottleneck is the SDR n-gram Tanimoto coefficient's inability to capture semantic similarity between topically related but lexically diverse records.

### Acceptance Criteria Evaluation

| Criterion | Result |
|-----------|--------|
| Practical concept coverage becomes non-zero | **FAIL** — 0% across all configs |
| Useful (not misleading) concepts | N/A — no concepts formed |
| No noticeable growth in false merges | **PASS** — 0 false merges everywhere |

## Recommendations

1. **Freeze Candidate C as STRUCTURALLY BLOCKED** — no further parameter tuning can help
2. **Keep ConceptSeedMode::Relaxed** as experimental opt-in — no harm, may help with future changes
3. **Keep TagFamily mode** as experimental opt-in — dramatically improves belief density (6 → 17)
4. **To unblock C in the future**, one of these architectural changes would be needed:
   - Replace SDR n-gram centroids with semantic embedding (e.g., sentence-level vectors)
   - Add a topic-model layer between beliefs and concepts
   - Use tag-based concept formation instead of SDR-based clustering
   - Relax concept clustering to tag-based similarity (shared tags ≥ 2) instead of centroid Tanimoto

## Files Changed

- **src/concept.rs**: `ConceptSeedMode` enum + `seed_mode` field + tag barrier in `cluster_beliefs()`
- **src/aura.rs**: `set_concept_seed_mode()` / `get_concept_seed_mode()` API
- **tests/concept_realistic_eval.rs**: 12 tests (realistic + dense + 4-way matrix) (NEW)
- **CONCEPT_REALISTIC_EVAL_REPORT.md**: this report (NEW)

## Test Summary

| Test | Status |
|------|--------|
| `realistic_corpus_standard_gates` | PASS |
| `realistic_corpus_relaxed_gates` | PASS |
| `realistic_corpus_standard_vs_relaxed` | PASS |
| `realistic_corpus_per_topic_detail` | PASS |
| `realistic_corpus_identity_stability` | PASS |
| `realistic_corpus_candidate_b_safe` | PASS |
| `realistic_corpus_cross_layer_intact` | PASS |
| `realistic_corpus_verdict` | PASS |
| `dense_corpus_positive_control` | PASS |
| `realistic_corpus_tagfamily_relaxed` | PASS |
| `four_way_comparison` | PASS |
| `dense_corpus_four_way` | PASS |

All 12 tests PASS. Zero failures.
