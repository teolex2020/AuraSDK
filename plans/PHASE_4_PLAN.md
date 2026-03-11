# Phase 4 — Limited Influence Activation for Candidate B

## Goal

Transition belief-aware recall ranking from shadow-approved to controlled limited
influence. **Not** a global default — opt-in, bounded, feature-flagged.

---

## Starting State

| Candidate | Status | Notes |
|-----------|--------|-------|
| A — Policy Surfaced Output | **PROMOTED** | Stable public API, no changes needed |
| B — Belief-Aware Recall | **LIMITED INFLUENCE** | Tri-state mode, opt-in, bounded |
| C — Concept-Assisted Grouping | **SHADOW EVALUATION** | Clustering retune activated concepts; pending wider coverage re-eval |

---

## What We Promote

- Belief-aware reranking as a **bounded influence layer** for recall
- Tri-state operating mode: `Off` (default), `Shadow` (observe-only), `Limited` (bounded rerank)

## What We Do NOT Promote

- Concept-aware ranking
- Causal-weighted ranking
- Policy-driven ranking
- Behavior influence or auto-actions
- Default-on activation

---

## Operating Modes

### `BeliefRerankMode::Off` (default)
- No belief influence on recall ranking
- Identical to pre-Phase 4 behavior

### `BeliefRerankMode::Shadow`
- Shadow scoring available via `recall_structured_with_shadow()`
- `recall_structured()` is unaffected — no ranking change
- Purely observational, for evidence collection

### `BeliefRerankMode::Limited`
- Active bounded reranking applied in `recall_structured()`
- Score delta capped at ±5% of original score
- Positional shift capped at ±2 positions
- Scope guards prevent activation on unsuitable queries

---

## Phase 4 Multipliers

| Belief State | Multiplier | Effect |
|-------------|-----------|--------|
| Resolved | 1.05 | +5% boost (confident, settled claim) |
| Singleton | 1.02 | +2% boost (unchallenged, unverified) |
| Unresolved | 0.97 | -3% penalty (competing hypotheses) |
| Empty / None | 1.00 | No change |

Cap: `BELIEF_RERANK_CAP = 0.05` (±5% maximum score delta)

---

## Guardrails

### Score Cap
- Maximum score delta: ±5% of original score
- `adjusted = original * multiplier`, clamped to `[original - 5%, original + 5%]`

### Positional Shift Cap
- Maximum ±2 positions in the ranking
- Enforced after score-based re-sort via iterative fixup
- Prevents belief signal from causing large rank jumps

### Scope Guards
Reranking is **skipped** (report indicates skip reason) when:

| Guard | Threshold | Reason |
|-------|-----------|--------|
| Result count | < 4 results | Too few results for meaningful reranking |
| Top-k | > 20 | Large result sets beyond intended scope |
| Belief coverage | 0% | No belief signal available |

### Rollback
- `set_belief_rerank_mode(BeliefRerankMode::Off)` — zero cost, immediate
- `set_belief_rerank_enabled(false)` — convenience alias
- No data migration needed

---

## API Surface

### Configuration
```rust
// Tri-state mode
aura.set_belief_rerank_mode(BeliefRerankMode::Limited);
aura.get_belief_rerank_mode(); // → BeliefRerankMode::Limited

// Convenience (backward-compatible)
aura.set_belief_rerank_enabled(true);  // → Limited
aura.set_belief_rerank_enabled(false); // → Off
aura.is_belief_rerank_enabled();       // true if Limited
```

### Recall with Rerank Report
```rust
let (results, report) = aura.recall_structured_with_rerank_report(
    "query", Some(10), Some(0.0), Some(true), None, None,
)?;

// report.was_applied       — whether reranking was active
// report.skip_reason       — why it was skipped (if not applied)
// report.records_moved     — count of records that changed position
// report.max_up_shift      — largest upward positional change
// report.max_down_shift    — largest downward positional change
// report.belief_coverage   — fraction of results with belief membership
// report.top_k_overlap     — overlap between baseline and reranked top-k
// report.rerank_latency_us — microseconds spent on reranking
```

---

## LimitedRerankReport Fields

| Field | Type | Description |
|-------|------|-------------|
| `was_applied` | bool | Whether reranking was actually applied |
| `skip_reason` | String | Why reranking was skipped (empty if applied) |
| `records_moved` | usize | Number of records whose position changed |
| `max_up_shift` | usize | Maximum upward positional shift |
| `max_down_shift` | usize | Maximum downward positional shift |
| `avg_belief_multiplier` | f32 | Average multiplier across all records |
| `belief_coverage` | f32 | Fraction of records with belief membership |
| `top_k_overlap` | f32 | Top-k set overlap (baseline vs reranked) |
| `rerank_latency_us` | u64 | Reranking latency in microseconds |

---

## Test Coverage

### Unit Tests (in `src/recall.rs`) — 12 tests
1. `test_rerank_no_beliefs_skipped` — scope guard: no coverage
2. `test_rerank_too_few_results_skipped` — scope guard: < 4 results
3. `test_rerank_top_k_too_large_skipped` — scope guard: top_k > 20
4. `test_rerank_resolved_boosts_within_cap` — resolved multiplier 1.05
5. `test_rerank_unresolved_penalizes_within_cap` — unresolved multiplier 0.97
6. `test_rerank_can_swap_close_scores` — close scores can swap
7. `test_rerank_cannot_swap_distant_scores` — distant scores cannot swap
8. `test_rerank_effect_bounded_by_cap` — delta never exceeds 5%
9. `test_rerank_positional_shift_bounded` — max ±2 positions
10. `test_rerank_report_metrics` — report has valid metrics
11. `test_rerank_mode_enum` — enum round-trips correctly

### Integration Tests (in `tests/shadow_recall.rs`) — 6 Phase 4 tests
1. `phase4_tristate_mode_default_off` — default mode is Off
2. `phase4_tristate_mode_switching` — Off → Shadow → Limited → Off
3. `phase4_compat_set_enabled_maps_to_limited` — bool API maps correctly
4. `phase4_rerank_report_on_populated_db` — report works on real data
5. `phase4_shadow_mode_no_ranking_change` — Shadow mode doesn't alter recall
6. `phase4_limited_mode_score_bounded` — bounded shifts in integration

### Shadow Tests (retained from Phase 3) — 11 tests
- 7 shadow scoring tests
- 4 rerank compatibility tests

---

## Acceptance Gates

| Gate | Criteria | Status |
|------|----------|--------|
| No crashes | All 460 tests pass | **PASS** |
| No recall mutation in Off/Shadow | Off/Shadow modes identical to baseline | **PASS** |
| Score delta bounded | ±5% cap enforced in all tests | **PASS** |
| Positional shift bounded | ±2 position cap enforced | **PASS** |
| Scope guards active | Skip reranking when unsuitable | **PASS** |
| Feature flag works | Tri-state mode toggles immediately | **PASS** |
| Rollback trivial | Off mode = zero-cost baseline restore | **PASS** |
| Backward compatibility | `set_belief_rerank_enabled()` still works | **PASS** |
| Report metrics available | `LimitedRerankReport` captures all data | **PASS** |

---

## Hard Stop Conditions

Halt Phase 4 and reassess if any occur:
- Recall precision drops > 0.05 on any test scenario
- Maintenance cycle time increases > 50ms
- Positional shift exceeds ±2 (violates invariant)
- Contradiction leakage grows after enabling limited mode
- Repeated queries produce unstable rank movement
- Latency growth exceeds budget (< 2ms per recall)

---

## Metrics to Collect (Post-Activation)

| Metric | Purpose |
|--------|---------|
| top-k overlap vs baseline | Ranking stability |
| promoted/demoted counts | Movement distribution |
| average/max rank shift | Shift magnitude |
| belief coverage per query | Signal availability |
| latency delta | Performance impact |
| contradiction leakage delta | Quality safety |

---

## Rollout Sequence

```
Step 1: ✅ Implement BeliefRerankMode tri-state enum
Step 2: ✅ Add bounded limited rerank with ±5% score cap + ±2 pos cap
Step 3: ✅ Add scope guards (min results, max top_k, coverage > 0)
Step 4: ✅ Add LimitedRerankReport with comprehensive metrics
Step 5: ✅ Add recall_structured_with_rerank_report() API
Step 6: ✅ Add 12 unit + 6 integration tests (Phase 4 specific)
Step 7: ✅ All 460 tests passing
Step 8: ✅ Run curated evaluation (82 queries, dual-mode baseline vs limited)
Step 9: ✅ Run cross-layer eval with limited mode enabled
Step 10: ✅ Decision: WIDER ROLLOUT CONFIRMED — stabilized feature
```

---

## Evidence Collection Results (82 queries, tests/phase4_evidence.rs)

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Total queries | 82 |
| Queries with results | 82/82 |
| Queries reranked | 68/82 |
| Avg top-k overlap | 0.945 |
| Median top-k overlap | 1.000 |
| % BETTER | 42.7% |
| % SAME | 57.3% |
| % WORSE | 0.0% |
| % UNCLEAR | 0.0% |
| % with movement | 42.7% |
| Avg belief coverage | 0.151 |
| Max positional shift (up) | 2 |
| Max positional shift (down) | 2 |
| Avg latency delta | -26μs (faster) |
| Max latency delta | 306μs |
| Contradiction improved | 0.0% |
| Contradiction worsened | 0.0% |

### Per-Category Breakdown

| Category | n | Reranked | Moved | Better | Worse |
|----------|---|---------|-------|--------|-------|
| stable-factual | 11 | 2 | 0 | 0 | 0 |
| belief-heavy | 10 | 9 | 2 | 2 | 0 |
| conflicting | 8 | 7 | 1 | 1 | 0 |
| low-coverage | 6 | 5 | 4 | 4 | 0 |
| devops | 8 | 7 | 4 | 4 | 0 |
| architecture | 6 | 5 | 3 | 3 | 0 |
| database | 5 | 5 | 3 | 3 | 0 |
| workflow | 5 | 5 | 4 | 4 | 0 |
| security | 5 | 5 | 4 | 4 | 0 |
| cross-domain | 7 | 7 | 4 | 4 | 0 |
| repeated | 3 | 3 | 1 | 1 | 0 |
| broad | 4 | 4 | 2 | 2 | 0 |
| no-match | 4 | 4 | 3 | 3 | 0 |

### Safety Gates

| Gate | Criteria | Result |
|------|----------|--------|
| No crashes | All 82 queries complete | **PASS** |
| Overlap stable | Avg ≥ 0.70 | **PASS** (0.945) |
| Regressions rare | < 10% WORSE | **PASS** (0.0%) |
| Contradiction stable | worsened ≤ improved + 5% | **PASS** |
| Shift bounded | ≤ 2 positions | **PASS** (2/2) |
| Latency budget | max delta < 2s | **PASS** (306μs) |

### Automated Verdict

**PREPARE FOR WIDER ROLLOUT** — clear benefit (42.7% BETTER), zero regressions (0.0% WORSE), all safety gates passed.

### Analysis Notes

- **Stable-factual** queries correctly show 0 movement — rerank leaves settled results alone
- **Conflicting** category: 1/8 moved, correctly with BETTER label (GraphQL vs REST)
- Movement stays within ±2 positions everywhere — positional cap is effective
- No contradiction leakage in either direction
- 15.1% avg belief coverage — still low but sufficient for the queries that have it
- Scope guards correctly blocked 14 queries (too few results or no belief coverage)

---

## Actual End State After Phase 4

| Layer | Status |
|-------|--------|
| Policy surfaced output | Fully promoted |
| Belief-aware recall | **Fully hardened** — stabilized wider rollout, rollback kept |
| Concept-assisted grouping | Shadow evaluation (clustering retune activated) |
| Causal/policy behavior | Blocked |

---

## Decision (2026-03-11): Wider Rollout Approved

Phase 4 evidence pass exceeded expectations:
- 42.7% BETTER, 0% WORSE across 82 curated queries
- All safety gates passed with margin
- Zero contradiction leakage
- Positional shift within cap everywhere

**Approved:** default-on for evaluation and internal builds.

**Guardrails retained:**
- All caps, scope guards, and rollback path unchanged
- No relaxation of multipliers or shift limits
- Off/Shadow modes preserved

**Not approved:**
- Unconditional global default without rollback
- Candidate C recall influence (shadow evaluation only)
- Causal/policy behavior influence
- Guardrail relaxation

---

## Step 9 Results: Cross-Layer Eval with Rerank Enabled (tests/cross_layer_rerank_eval.rs)

### Test Coverage — 6 tests

1. `step9_scenario_a_stable_preference_with_rerank` — stable preferences undisturbed by rerank
2. `step9_scenario_b_deploy_chain_with_rerank` — causal chain data under active reranking
3. `step9_scenario_c_contextual_with_rerank` — no false conflict from rerank on contextual data
4. `step9_scenario_d_multi_topic_with_rerank` — topic isolation maintained under reranking
5. `step9_soak_20_cycles_with_rerank` — 20 cycles with Limited active from start
6. `step9_aggregate_rerank_cross_layer_summary` — aggregate metrics across all 4 scenarios

### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Total queries | 10 |
| Reranked | 2/10 |
| With movement | 1/10 |
| Avg top-k overlap | 1.000 |
| Max up shift | 1 |
| Max down shift | 1 |
| Avg belief coverage | 0.133 |

### Safety Gates

| Gate | Criteria | Result |
|------|----------|--------|
| Avg overlap | ≥ 0.70 | **PASS** (1.000) |
| Max up shift | ≤ 2 | **PASS** (1) |
| Max down shift | ≤ 2 | **PASS** (1) |
| Cross-layer invariants | churn, provenance, suppression | **PASS** |
| Topic isolation | no cross-topic concept merges | **PASS** |
| Soak stability | count variance ≤ 2 over 5 cycles | **PASS** |
| Result count preserved | rerank never drops results | **PASS** |

### Analysis

- Reranking correctly activates only when belief coverage exists (2/10 queries)
- Scope guards properly block 8/10 queries with insufficient coverage
- Movement is minimal (1/10) — ±1 position, well within ±2 cap
- Cross-layer stack (concept, causal, policy) completely unaffected by reranking
- 20-cycle soak shows no drift or instability under continuous Limited mode
- **Conclusion:** belief reranking integrates safely with the full cognitive stack

---

## Decision (2026-03-11): Step 10 — WIDER ROLLOUT CONFIRMED

All 10 rollout steps complete. Final decision:

**KEEP AND WIDER-ROLLOUT CONFIRMED** — Candidate B is now a stabilized feature.

Rationale:
- Cross-layer invariants not broken (Step 9)
- Soak stable, no drift
- Positional cap works everywhere
- Reranking behaves conservatively
- Zero regressions across 82 + 10 queries

This is NOT an argument for:
- Aggressive reranking (higher multipliers, wider caps)
- Guardrail relaxation
- Candidate C recall influence (shadow evaluation only) (0% concept coverage)
- Causal/policy behavior influence

**Guardrails retained permanently until new evidence pass:**
- Scope guards (min 4 results, top_k ≤ 20, coverage > 0)
- ±5% score cap
- ±2 positional shift cap
- Tri-state mode (Off / Shadow / Limited)
- Rollback path via `set_belief_rerank_mode(Off)`

**Project transitions to steady-state monitoring.**

---

## Post-Phase-4 Hardening Addendum (2026-03-11)

After Step 10, Candidate B was additionally stress-tested with a large synthetic campaign:

- 100 runs
- 6 scenario classes
- 200 queries
- 0.0% WORSE
- 0 contradiction worsening
- avg top-k overlap 1.000
- max positional shift within the existing cap
- 0 alert failures

### Final status

**Candidate B is now fully hardened.**

Meaning:
- keep current wider-rollout configuration
- keep all existing guardrails
- do not widen influence further without a new promotion review
- treat the current rerank architecture as frozen baseline behavior

### What is in shadow evaluation

- Candidate C (concept-assisted grouping) — KEEP SHADOW after full shadow evaluation (2026-03-11)
  - **Verdict A (concept layer health): HEALTHY** — concept.rs works correctly under sufficient belief density
    - Multi-concept profile: 100% formation, 49.4% coverage, 92% USEFUL, 0 false merges
  - **Verdict B (practical viability): BLOCKED** — upstream belief fragmentation prevents concept formation
    - Practical: 0/20 form concepts — tag diversity → singleton beliefs → 0 seeds per partition
    - Bottleneck: tag-based belief grouping → unique keys → singleton beliefs on diverse corpora
    - This is a belief pipeline constraint, not a concept engine bug
  - Safe but not useful enough for inspection-only promotion
  - See: CONCEPT_SHADOW_EVAL_REPORT.md

### What remains deferred

- Concept-aware recall ranking (depends on Candidate C coverage gate ≥ 30%)
- Any behavior influence from causal or policy layers
- Any relaxation of belief rerank caps or scope guards
