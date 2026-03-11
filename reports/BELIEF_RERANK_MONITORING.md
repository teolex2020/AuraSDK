# Candidate B - Belief Rerank Steady-State Monitoring

Status: active - monitoring mode, not feature development.

## Purpose

Operational monitoring for the belief reranking feature (Candidate B) in stabilized wider rollout. Tracks health metrics, detects regressions, and triggers alerts when thresholds are violated.

---

## Running the Monitor

```bash
cargo test --no-default-features --features "encryption,server,audit" \
  --test belief_rerank_monitor -- --nocapture
```

Recommended cadence: weekly, or after any code change touching recall, belief, or maintenance.

---

## Monitoring Query Pack

28 stable queries across 10 categories. The pack is fixed to enable trend comparison across runs.

| Category | Queries | Purpose |
|----------|---------|---------|
| stable-factual | 4 | Baseline - should be SAME |
| belief-heavy | 4 | Where reranking adds value |
| conflicting | 4 | Unresolved beliefs - penalize |
| devops | 3 | Mixed belief coverage |
| architecture | 3 | Domain knowledge |
| database | 2 | Preference + fact mix |
| workflow | 2 | Personal preferences |
| security | 2 | Decision-heavy |
| cross-domain | 2 | Mixed signals |
| no-match | 2 | Edge cases - should skip |

---

## Metrics Collected

| Metric | Description |
|--------|-------------|
| queries_reranked | How many queries had reranking applied |
| skip_rate | Percent of queries where scope guards blocked reranking |
| avg_top_k_overlap | Overlap between baseline (Off) and Limited top-k sets |
| pct_better | Percent of queries where reranking improved quality |
| pct_same | Percent of queries with no change |
| pct_worse | Percent of queries where reranking degraded quality |
| pct_unclear | Percent of queries with ambiguous quality impact |
| pct_with_movement | Percent of queries with any positional movement |
| avg_belief_coverage | Average fraction of results with belief membership |
| avg_latency_delta_us | Average latency difference (Limited - Baseline) |
| max_latency_delta_us | Worst-case latency difference |
| pct_contradiction_worsened | Percent of queries where unresolved records rose in rank |
| max_up_shift / max_down_shift | Largest observed positional shifts |

---

## Alert Thresholds

| Alert | Threshold | Severity | Action |
|-------|-----------|----------|--------|
| WORSE rate | <= 5% | CRITICAL | Investigate immediately; consider rollback to Off |
| Avg top-k overlap | >= 0.70 | CRITICAL | Reranking too aggressive; review multipliers |
| Max positional shift | <= 2 | CRITICAL | Hard cap violation; indicates code bug |
| Avg latency delta | <= 2000us | HIGH | Performance regression; profile rerank path |
| Contradiction worsened | <= 5% | HIGH | Unresolved records rising in rank; review multipliers |
| Belief coverage | >= 1% | MEDIUM | Low coverage means reranking rarely activates |
| Scope-guard skip rate | <= 95% | MEDIUM | Too many skips means coverage problem |

### Interpretation

- CRITICAL alerts -> test fails, investigate before next run
- HIGH alerts -> test fails, likely needs code investigation
- MEDIUM alerts -> test passes with warning, track trend over time

---

## Baseline (First Run - 2026-03-11)

| Metric | Value |
|--------|-------|
| Total queries | 28 |
| Reranked | 13/28 |
| Skip rate | 53.6% |
| Avg overlap | 0.939 |
| Percent BETTER | 25.0% |
| Percent SAME | 75.0% |
| Percent WORSE | 0.0% |
| Avg coverage | 7.9% |
| Avg latency delta | -48us |
| Max shift | up 2 / down 2 |
| Alerts | ALL PASS |

---

## Trend Log

Record each monitoring run here for trend tracking.

| Date | Reranked | Overlap | Better | Worse | Coverage | Latency | Alerts |
|------|----------|---------|--------|-------|----------|---------|--------|
| 2026-03-11 | 13/28 | 0.939 | 25.0% | 0.0% | 7.9% | -48us | ALL PASS |

### Synthetic Stress Pass (10 varied runs - 2026-03-11)

Purpose:
- validate that bounded belief reranking remains stable across small deterministic corpus and query variations
- catch nondeterminism, hidden drift, or cap violations earlier than periodic fixed-pack monitoring

Method:
- run [tests/belief_rerank_varied_monitor.rs](/d:/AuraSDK-verify/tests/belief_rerank_varied_monitor.rs)
- 10 deterministic synthetic runs with varied phrasing
- evaluate the same safety gates used by the steady-state monitor

Summary:

| Metric | Value |
|--------|-------|
| Runs | 10 |
| Avg reranked | 100.0% |
| Avg BETTER | 11.2% |
| Avg SAME | 88.8% |
| Avg WORSE | 0.0% |
| Avg overlap | 0.959 |
| Avg coverage | 91.6% |
| Avg latency delta | -14us |
| Max shift observed | up 1 / down 2 |
| Alerts | ALL PASS |

Interpretation:
- reranking remained bounded under synthetic variation
- no regressions were observed
- positional cap held in every run
- latency stayed effectively flat

This increases engineering confidence, but does not replace the fixed-pack monitor used for operational trend tracking.

---

## Rollback Procedure

If any CRITICAL alert fires:

1. Set mode to Off: `aura.set_belief_rerank_mode(BeliefRerankMode::Off)`
2. Re-run monitor to confirm baseline behavior restored
3. Investigate root cause
4. Fix and re-run evidence pass before re-enabling

Rollback is instant, zero-cost, no data migration needed.

---

## Candidate C Re-Check Cadence

Separate from Candidate B monitoring. Run concept coverage eval periodically:

```bash
cargo test --no-default-features --features "encryption,server,audit" \
  --test concept_coverage_eval -- --nocapture
```

Gate: coverage >= 30%. Current: 0%. Do not implement Candidate C until gate passes.

Recommended cadence: monthly, or when corpus significantly grows.

---

## What This Monitor Does NOT Do

- Does not test with real user data (uses fixed corpus)
- Does not measure production latency under load
- Does not test concurrent access patterns
- Does not replace integration tests (`cross_layer_eval`, `cross_layer_rerank_eval`)

For full validation after code changes, run the complete test suite:

```bash
cargo test --no-default-features --features "encryption,server,audit"
```
