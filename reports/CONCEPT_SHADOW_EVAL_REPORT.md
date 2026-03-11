# Concept Shadow Evaluation Report

**Date:** 2026-03-11
**Sprint:** Concept Shadow Evaluation
**Harness:** `tests/concept_shadow_eval.rs`
**Config:** CONCEPT_SIMILARITY_THRESHOLD=0.10, MIN_BELIEF_STABILITY=2.0, MIN_BELIEF_CONFIDENCE=0.55

## DUAL VERDICT

**Verdict A (concept layer health): HEALTHY** — concept.rs works correctly under sufficient belief density

**Verdict B (practical viability): BLOCKED** — upstream belief fragmentation prevents concept formation on current corpora

These are independent findings. The concept engine itself is sound; the bottleneck is in the belief formation pipeline, not in concept discovery or clustering.

---

## Track A: Belief-Density-Aware Synthetic Pack (60 runs, 6 profiles, 8 cycles/run)

| Profile | n | With Concepts | Concepts | Stable | Avg Coverage | Useful% | Empty% | False Merges |
|---------|---|:---:|---:|---:|---:|---:|---:|---:|
| single-stable | 10 | 0 | 0 | 0 | 0.0% | 0.0% | 100.0% | 0.00 |
| core-shell | 10 | 0 | 0 | 0 | 0.0% | 0.0% | 100.0% | 0.00 |
| two-nearby | 10 | 0 | 0 | 0 | 0.0% | 0.0% | 100.0% | 0.00 |
| **multi-concept** | **10** | **10** | **10** | **10** | **49.4%** | **92.0%** | **0.0%** | **0.00** |
| mixed-topic | 10 | 0 | 0 | 0 | 0.0% | 0.0% | 100.0% | 0.00 |
| sparse-adversarial | 10 | 0 | 0 | 0 | 0.0% | 0.0% | 100.0% | 0.00 |

### Synthetic Global Metrics

| Metric | Value |
|--------|-------|
| Total runs | 60 |
| Runs with concepts | 10/60 (16.7%) |
| Total concepts formed | 10 |
| Total stable | 10 |
| Avg coverage | 12.8% |
| % queries with concept | 25.0% |
| Utility USEFUL | 25.0% |
| Utility NEUTRAL | 2.0% |
| Utility MISLEADING | 0.0% |
| Utility EMPTY | 75.0% |
| False merge rate | 0.00 |

### Why Only multi-concept Profile Forms Concepts

The concept pipeline requires this chain to activate:
1. Multiple records with **identical tag sets** → grouped into same belief
2. Beliefs reach **stability >= 2.0** (requires multiple maintenance cycles)
3. Beliefs reach **confidence >= 0.55** (pass seed gate)
4. **>= 2 seeds in the same (namespace, semantic_type) partition**
5. SDR centroid Tanimoto **>= 0.10** between seed pairs

Most profiles fail at steps 1-4:

- **single-stable**: 10 records, same tags → 1 belief → only 1 seed → no clustering possible
- **core-shell**: 2 tag groups → 2-3 beliefs, but partitioned by different tag sets → 0 pairs in same partition
- **two-nearby**: 2 tag groups → 2+ beliefs, different partitions (`deploy,safety` vs `deploy,speed`) → 0 pairs
- **multi-concept**: 4 families, each with 5 records and identical tags → 4+ beliefs in same partition (`decision`) → clustering finds pairs above 0.10
- **mixed-topic**: 4 topics with 3 records each, different semantic_types (`decision` vs `fact`) → beliefs split across partitions
- **sparse-adversarial**: 4 records, all different tags → 0 beliefs formed

The critical insight: **only the multi-concept profile has enough records with identical tags sharing the same semantic_type** to produce multiple beliefs in one partition.

---

## Track B: Diagnostic Negative Control — Practical Query Sets (20 runs, 4 scenarios, 8 cycles/run)

Track B is not a fairness gate for concept.rs — it is a **diagnostic negative control** confirming that the upstream belief pipeline does not produce enough partition density on realistic corpora for concepts to emerge. This is expected behavior, not a concept engine failure.

| Scenario | n | With Concepts | Concepts | Coverage | Useful% | Empty% |
|----------|---|:---:|---:|---:|---:|---:|
| stable-preference | 5 | 0 | 0 | 0.0% | 0.0% | 100.0% |
| deploy-chain | 5 | 0 | 0 | 0.0% | 0.0% | 100.0% |
| multi-topic | 5 | 0 | 0 | 0.0% | 0.0% | 100.0% |
| contextual | 5 | 0 | 0 | 0.0% | 0.0% | 100.0% |

### Practical Global: 0 concepts across all 20 runs

Practical scenarios use small diverse corpora (4-6 records) with varied tags and semantic types. This produces 0-2 beliefs total, 0-1 seeds, and never reaches the ">=2 seeds in same partition" threshold required for concept formation.

---

## Safety Metrics

| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Recall degraded | 0 | 0 | **PASS** |
| Identity unstable | 0 | 0 | **PASS** |
| Provenance gaps | 0 | 0 | **PASS** |
| Misleading rate | < 5% | 0.0% | **PASS** |
| False merge rate | < 5% | 0.00 | **PASS** |

All safety gates pass. The concept layer is completely safe — it just doesn't produce signal on most corpora.

---

## Utility Label Distribution (200 queries total)

| Label | Count | % | Description |
|-------|-------|---|-------------|
| USEFUL | 46 | 23.0% | Concepts group related results meaningfully |
| NEUTRAL | 4 | 2.0% | Concepts exist but add minimal value |
| MISLEADING | 0 | 0.0% | Concepts would create false abstractions |
| EMPTY | 150 | 75.0% | No concept coverage |

All USEFUL labels come from the multi-concept synthetic profile, which has the ideal structure for concept formation.

---

## Gate Assessment — Verdict A (Concept Layer Health)

These gates evaluate whether concept.rs works correctly when given sufficient belief density (multi-concept profile).

| Gate | Criteria | Result | Status |
|------|----------|--------|--------|
| Safety | No recall degrade, misleading < 5%, false merge < 5% | All pass | **PASS** |
| Dense coverage | >= 30% on multi-concept | 49.4% | **PASS** |
| Dense USEFUL | >= 50% on multi-concept | 92.0% | **PASS** |
| Dense no misleading | < 5% misleading | 0.0% | **PASS** |
| Identity stable | 0 unstable across replay | 0 | **PASS** |

**Verdict A: HEALTHY** — concept.rs works correctly under sufficient belief density.

## Gate Assessment — Verdict B (Practical Viability)

These gates evaluate whether current corpora produce enough belief density for concepts to emerge in practice.

| Gate | Criteria | Result | Status |
|------|----------|--------|--------|
| Global synthetic coverage | >= 15% | 12.8% | **FAIL** |
| Practical has concepts | > 0 runs | 0/20 | **FAIL** |
| Practical coverage | >= 5% | 0.0% | **FAIL** |
| Inspection-ready | synth_cov >= 25% AND useful >= 10% AND practical > 0 | Not met | **FAIL** |

**Verdict B: BLOCKED** — upstream belief fragmentation prevents concept formation on current corpora. This is not a concept engine bug — it is a belief pipeline constraint.

---

## Structural Analysis

### The Upstream Bottleneck

The concept layer itself is healthy. The bottleneck is in the **belief formation pipeline**:

```
Records → [tag-based grouping] → Beliefs → [seed gate] → Seeds → [SDR clustering] → Concepts
              ↑                        ↑
        TAG DIVERSITY HERE       STABILITY GATE HERE
        causes singletons        filters out young beliefs
```

For concepts to form, the corpus must satisfy ALL of these simultaneously:
1. **Tag repetition**: Multiple records must share the exact same sorted tag set
2. **Stability**: Beliefs must survive multiple maintenance cycles (stability >= 2.0)
3. **Partition density**: >= 2 qualifying beliefs must share the same `(namespace, semantic_type)`
4. **Centroid similarity**: SDR centroids of paired beliefs must have Tanimoto >= 0.10

Real-world corpora tend to have diverse tags (each record describes a slightly different facet), which means step 1 fails — each record gets its own unique belief key and stays as a Singleton.

### What Would Fix This

Without changing scope (no recall influence, no compression):

1. **SDR-based belief subclustering** already exists in the belief engine, but the coarse step (tag-key matching) prevents it from grouping records with different but related tag sets
2. **Relaxing the seed gate** (lower MIN_BELIEF_STABILITY or MIN_BELIEF_CONFIDENCE) would let more beliefs pass but risks promoting unstable beliefs
3. **Natural corpus growth** — as users store more records on the same topics, beliefs will naturally accumulate identical tag sets and reach stability thresholds

None of these changes are in scope for this sprint.

---

## Inspection-Only Readiness Check

| Criterion | Assessment |
|-----------|-----------|
| Can concept labels be shown without misleading users? | Yes — 0% misleading rate |
| Does grouping hide record-level detail? | No — concepts are supplementary, not replacing records |
| Is provenance complete? | Yes — all concepts have belief_ids + record_ids |
| Do concepts create false authority? | No — but they also provide no signal on practical data |

**Assessment:** Concepts are **safe** for inspection but **not useful** on practical data. Showing them would display nothing in most real scenarios, which is technically harmless but not worth building a UI for.

---

## Test Inventory

| Test | Status |
|------|--------|
| concept_shadow_eval_mixed_pack_runs | PASS |
| concept_shadow_eval_practical_pack_runs | PASS |
| concept_shadow_eval_zero_recall_impact | PASS |
| concept_shadow_eval_identity_stable_on_replay | PASS |
| concept_shadow_eval_no_provenance_gaps | PASS |
| concept_shadow_eval_no_cross_topic_explosion | PASS |
| concept_shadow_eval_emits_final_verdict | PASS |

All 7 tests pass.

---

## Conclusion

### What Works
- Clustering retune (0.20 → 0.10) correctly activates concepts on focused corpora
- Zero false merges, zero recall degradation, deterministic identity
- Multi-concept profile: 100% concept formation, 49.4% coverage, 92% USEFUL
- Safety is perfect across all 80 runs

### What Doesn't Work Yet
- 5/6 synthetic profiles produce 0 concepts (insufficient belief density per partition)
- All 4 practical scenarios produce 0 concepts (diverse tags → singleton beliefs)
- Coverage gate (15%) fails globally at 12.8%
- Practical coverage is 0%

### Combined Verdict: KEEP SHADOW

**Verdict A (concept layer health): HEALTHY** — concept.rs works correctly under sufficient belief density. Multi-concept profile: 100% formation, 49.4% coverage, 92% USEFUL, 0 false merges, deterministic identity.

**Verdict B (practical viability): BLOCKED** — upstream belief fragmentation prevents concept formation on current corpora. Tag diversity → unique belief keys → singleton beliefs → 0 seeds per partition.

These are independent findings. The concept engine itself is sound; the practical rollout is blocked by the belief formation pipeline, not by concept discovery or clustering.

### Path Forward (out of scope, documented for future reference)
1. Wait for natural corpus growth (more records with overlapping tags)
2. Consider enhancing belief subclustering to merge records with related (not identical) tag sets
3. Re-evaluate when belief density improves
4. Do NOT force concept formation by relaxing safety thresholds
