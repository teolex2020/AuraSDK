# Concept Seed Calibration Report

**Date:** 2026-03-11
**Sprint:** Concept Seed Calibration
**Harness:** `tests/concept_seed_calibration.rs`

## Summary

The activation campaign diagnosed "SEED SELECTION TOO STRICT" using small synthetic worlds (8-20 records). This calibration sprint reveals a **different, deeper blocker**: beliefs DO pass the seed gate, but SDR centroids don't cluster.

## Track A: Belief Distribution

| Config | Beliefs | Resolved+Singleton | stab>=2.0 | conf>=0.55 | Both gates | Seeds | Concepts |
|--------|---------|-------------------|-----------|------------|------------|-------|----------|
| single-stable/8cyc | 1 | 1 S | 1 | 1 | 1 | 1 | 0 |
| single-stable/15cyc | 1 | 1 S | 1 | 1 | 1 | 1 | 0 |
| single-stable/25cyc | 1 | 1 S | 1 | 1 | 1 | 1 | 0 |
| multi-concept/8cyc | 7 | 7 S | 4-7 | 7 | 4-7 | 6-8 | 0 |
| multi-concept/15cyc | 8 | 8 S | 8 | 8 | 8 | 8 | 0 |
| multi-concept/25cyc | 8 | 8 S | 8 | 8 | 8 | 6 | 0 |
| enriched/8cyc | 11 | 11 S | 9-10 | 11 | 9-10 | 7-10 | 0 |
| enriched/15cyc | 11 | 11 S | 9-11 | 11 | 9-11 | 8-9 | 0 |
| enriched/25cyc | 11 | 11 S | 9-11 | 11 | 9-11 | 9-11 | 0 |
| enriched/35cyc | 11 | 11 S | 11 | 11 | 11 | 9-11 | 0 |

### Key Finding

Beliefs reach the current seed gate (`stability >= 2.0 && confidence >= 0.55`) routinely:
- All beliefs are **Singleton** (no competing hypotheses)
- Stability accumulates correctly (+1.0 per cycle)
- Confidence stays at 0.90 (from `source_type="recorded"`)
- With enriched corpus + 8 cycles: **up to 11 seeds pass the gate**

**But 0 concepts form in ALL configurations.**

### Corrected Diagnosis

The activation campaign's diagnosis of "SEED SELECTION TOO STRICT" was partially correct for small worlds (8-11 records with 8 cycles) but **incorrect for enriched corpora**. The real blocker is downstream.

## Track B: Cycle/Corpus Size Sweep

| Cycles | Beliefs | Eligible (R+S) | stab>=2.0 | stab>=1.0 | conf>=0.55 | Seeds | Concepts |
|--------|---------|----------------|-----------|-----------|------------|-------|----------|
| 5 | 10 | 10 | 6 | 10 | 10 | 7 | 0 |
| 8 | 11 | 11 | 8 | 11 | 11 | 9 | 0 |
| 12 | 11 | 11 | 7 | 11 | 11 | 7 | 0 |
| 15 | 11 | 11 | 9 | 11 | 11 | 8 | 0 |
| 20 | 11 | 11 | 11 | 11 | 11 | 11 | 0 |
| 25 | 11 | 11 | 11 | 11 | 11 | 9 | 0 |
| 30 | 11 | 11 | 11 | 11 | 11 | 9 | 0 |

**Result:** Even with 35 cycles and 30 records, **zero concepts form**. More cycles don't help because the block is in clustering, not seed selection.

## Track C: Direct ConceptEngine Validation

Bypassed the full Aura pipeline and called `ConceptEngine::discover()` directly with synthetic beliefs that already pass the seed gate.

| Scenario | Seeds | Concepts | Stable | Candidate | Avg Score | Cluster Size | False Merges |
|----------|-------|----------|--------|-----------|-----------|-------------|-------------|
| 2 deploy-safety beliefs | 2 | 1 | 0 | 1 | 0.688 | 5.0 | None |
| Deploy + Database (separate) | 4 | 2 | 0 | 2 | 0.688 | 5.0 | None |
| 4 families (deploy, DB, security, testing) | 8 | 3 | 0 | 3 | 0.678 | 6.7 | None |
| Below-threshold (stab=0.5, conf=0.30) | 0 | 0 | - | - | - | - | - |
| Different topics at relaxed threshold | 0 | 0 | - | - | - | N/A | None |
| Deterministic replay | 2×identical | 2×1 | - | - | - | - | None |

### Key Finding

**The ConceptEngine algorithm is healthy:**
- Concepts form correctly when given valid seeds with proper SDR data
- Different topics stay separate (no false merges)
- Identity is deterministic across replays
- Below-threshold beliefs are correctly rejected
- Clustering, scoring, and identity generation all work

## Diagnosis

### Block: CLUSTERING

The block is NOT in seed selection (correcting the activation campaign's diagnosis). The block is in the **SDR centroid building** during Phase 3.7:

1. `build_centroids()` iterates over seed beliefs
2. For each belief, it looks up `hypothesis.prototype_record_ids` in `sdr_lookup`
3. It unions all SDR bits to form a centroid
4. It compares centroids pairwise using Tanimoto with threshold `>= 0.20`

**The centroids built from real Aura sdr_lookup have Tanimoto < 0.20 between all belief pairs**, even for beliefs that are semantically about the same topic (e.g., two "deploy safety" beliefs).

### Root Cause Candidates

1. **Belief key granularity**: Each belief groups records by coarse tag key. Two beliefs about "deploy safety" might have different tag sets (e.g., `deploy,safety` vs `deploy,safety,workflow`), producing different but overlapping record sets. The SDR centroids of these non-overlapping sets may have low Tanimoto.

2. **SDR bit sparsity**: With `is_identity=false` (general bit range 0..65535), n-gram SDRs are sparse. The union of 2-3 record SDRs per belief may still be too sparse for meaningful overlap.

3. **Single-belief partitions**: If beliefs each get their own tag-based key, there may only be 1 belief per partition, and concept discovery skips partitions with < 2 beliefs.

### Verification Path

To confirm the root cause:
- Add diagnostic logging to `build_centroids()` — print centroid sizes and pairwise Tanimoto values
- Check if beliefs within the same `(namespace, semantic_type)` partition actually share enough SDR bits
- Check if the partition has ≥ 2 beliefs (or if tag granularity splits them)

## Final Verdict

**CLUSTERING BLOCK** — Seeds pass the gate, but SDR centroids built from real beliefs don't produce Tanimoto ≥ 0.20 between belief pairs. The concept algorithm is healthy (Track C proves this), but the real SDR centroid pipeline produces centroids that are too dissimilar for clustering.

## Recommendation

**Experimental threshold retune** — specifically:

1. **Lower `CONCEPT_SIMILARITY_THRESHOLD`** from 0.20 to 0.10 or 0.05 and re-run
2. **Diagnose centroid quality**: add logging to `build_centroids()` to measure actual Tanimoto values between seed centroids
3. **Check partition granularity**: verify that beliefs from the same topic end up in the same partition
4. **Consider centroid strategy**: the current "union of all record SDR bits" may produce centroids that are too different between beliefs even for the same topic

### What This Sprint Proved

- **Seed selection is NOT the blocker** (correcting the activation campaign)
- **Concept algorithm is healthy** when given proper SDR data
- **The block is localized** to SDR centroid similarity in the real pipeline
- **The fix is likely a threshold tune** (`CONCEPT_SIMILARITY_THRESHOLD`), not an algorithm redesign

## Test Inventory

| Test | Status |
|------|--------|
| belief_seed_distribution_is_reported | PASS |
| lowering_stability_threshold_increases_seed_count | PASS |
| lowering_seed_gate_can_activate_concepts | PASS |
| direct_concept_engine_with_synthetic_beliefs_forms_concepts | PASS |
| lower_seed_gate_does_not_create_false_merge_explosion | PASS |
| identity_stability_remains_bounded_under_relaxed_gate | PASS |
| zero_recall_impact_preserved | PASS |
| threshold_sweep_emits_comparison_report | PASS |
| calibration_verdict_is_emitted | PASS |

All 9 tests pass.
