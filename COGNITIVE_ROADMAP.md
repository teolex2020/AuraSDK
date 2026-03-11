# Aura Cognitive Roadmap

Status: internal roadmap for controlled evolution of Aura's cognitive stack.
Date: 2026-03-11 (updated)

Detailed sprint reports now live under [reports/](/d:/AuraSDK-verify/reports) and promotion plans under [plans/](/d:/AuraSDK-verify/plans).

## Purpose

This document defines a safe roadmap for evolving Aura from a strong memory engine into a fuller cognitive engine without breaking the project's current strengths:

- deterministic local recall
- low latency
- zero-LLM maintenance
- offline operation
- simple integration surface

The target direction is:

`Record -> Belief -> Concept -> Causal Pattern -> Policy`

This is not a pivot away from Aura's current architecture. It is the natural extension of what already exists in the codebase.

## Current State

Aura now has the full 5-layer hierarchy implemented in advisory/read-only form:

- `Record` is mature in [src/record.rs](/d:/AuraSDK-verify/src/record.rs)
- `Belief` exists in [src/belief.rs](/d:/AuraSDK-verify/src/belief.rs)
- `Concept` exists in [src/concept.rs](/d:/AuraSDK-verify/src/concept.rs)
- `Causal Pattern` exists in [src/causal.rs](/d:/AuraSDK-verify/src/causal.rs)
- `Policy` exists in [src/policy.rs](/d:/AuraSDK-verify/src/policy.rs)
- `run_maintenance()` executes all cognitive phases in [src/aura.rs](/d:/AuraSDK-verify/src/aura.rs)
- maintenance reports expose epistemic, belief, concept, causal, and policy metrics through [src/background_brain.rs](/d:/AuraSDK-verify/src/background_brain.rs)

### Current Operating Mode

All higher layers currently follow the same safety model:

- full rebuild each maintenance cycle
- derived state, not source of truth
- write-only cache stores for inspection
- zero recall impact by design
- zero automatic behavior impact by design
- full provenance back to lower-layer IDs

### Existing Strengths To Preserve

- Recall remains fast and deterministic.
- Existing APIs keep working unchanged.
- Maintenance remains safe on empty and large brains.
- New cognitive layers remain advisory-first.
- No new dependency on external models or hosted services is introduced.

## Layer Status

### 1. Record

Role:
- atomic episodic memory unit
- carries retention, provenance, semantic role, and epistemic state

Implemented:
- `strength`
- `activation_count`
- `decay`
- `connections`
- `semantic_type`
- `source_type`
- `namespace`
- `confidence`
- `support_mass`
- `conflict_mass`
- `volatility`

Current state:
- operational
- integrated into maintenance
- belief state now influences recall ranking via bounded reranking (Phase 4, wider rollout)

Primary upside:
- separates "frequently activated" from "likely true"

Primary risk:
- poor heuristics can bias higher layers if support/conflict propagation gets too noisy

### 2. Belief

Role:
- aggregate records into competing hypotheses about the same claim
- maintain resolved vs unresolved epistemic state

Implemented:
- SDR-based claim grouping
- hypothesis scoring
- `resolved`, `unresolved`, `singleton`, `empty`
- churn metrics
- persistence in `beliefs.cog`
- replay, soak, and benchmark coverage

Current state:
- hardened
- **wider rollout**: belief state influences recall ranking via bounded rerank (±5% score, ±2 positions)
- tri-state mode: Off / Shadow / Limited (default-on for eval/internal)
- evidence: 82 queries, 42.7% BETTER, 0% WORSE, all safety gates passed
- synthetic campaign: 100 runs, 200 queries, 0% WORSE, 0 alert failures

Primary upside:
- system stops treating all conflicting records as equal truths
- resolved beliefs subtly boost relevant recall results

Primary risk:
- false grouping still remains the main failure mode if future data distributions shift

### 3. Concept

Role:
- compress repeated belief patterns into reusable abstractions
- represent stable core vs variable shell

Implemented:
- dynamic concept discovery in [src/concept.rs](/d:/AuraSDK-verify/src/concept.rs)
- partitioned clustering by `namespace` and `semantic_type`
- stable concept identity based on abstraction features
- read-only derived-state rebuild each cycle
- `concepts.cog` as inspection cache only
- centroid diagnostics in ConceptReport/ConceptPhaseReport

Current state:
- **KEEP SHADOW** — shadow evaluation completed (2026-03-11)
- Clustering retune (0.20 → 0.10) activated concepts on focused synthetic corpora
- Shadow eval (80 runs): multi-concept profile 49.4% coverage, 92% USEFUL; all other profiles 0%
- Practical scenarios: 0 concepts — upstream belief density insufficient (tag diversity → singleton beliefs)
- Bottleneck: tag-based belief grouping produces unique keys on diverse corpora → 0-1 seeds per partition
- Safety perfect: 0% misleading, 0 false merges, 0 recall degradation, deterministic identity
- not yet trusted for compression or recall influence

Primary upside:
- reusable abstractions over repeated stable beliefs

Primary risk:
- false abstraction if grouping quality degrades on new data

### 4. Causal Pattern

Role:
- learn directed relationships of the form condition -> event -> outcome
- move beyond similarity into consequence modeling

Implemented:
- explicit causal edges
- temporal ordering within bounded window
- belief-level aggregation
- causal scoring and classification
- hardening tests for confounders and temporal noise
- `causal.cog` as write-only inspection cache

Current state:
- hardened for advisory use
- not yet trusted as downstream control input

Primary upside:
- operational learning from repeated consequences

Primary risk:
- correlation promoted to causation under new or noisy workloads

### 5. Policy

Role:
- convert learned beliefs, concepts, and causal patterns into behavioral hints

Implemented:
- advisory policy hint discovery in [src/policy.rs](/d:/AuraSDK-verify/src/policy.rs)
- deterministic action templates
- provenance to causal, concept, belief, and record IDs
- suppression for competing hints
- `policies.cog` as write-only inspection cache

Current state:
- read-only advisory layer only
- suitable for inspection and surfaced recommendations later
- not suitable for auto-action or behavior control

Primary upside:
- first user-visible layer that can explain "what the system recommends"

Primary risk:
- fake authority if weak upstream patterns are surfaced too aggressively

## Completed Phases

### Phase 0: Stabilize Epistemic Base

Status:
- complete

Delivered:
- record epistemic fields
- epistemic maintenance phase
- belief observability
- replay and soak tests

### Phase 0.5: Harden Belief Formation

Status:
- complete

Delivered:
- SDR-based claim grouping
- grouping benchmark gates
- churn hardening
- contradiction scoping
- replay stability and curated benchmark coverage

### Phase 1: Read-Only Cognitive Stack

Status:
- complete

Delivered:
- `concept.rs`
- `causal.rs`
- `policy.rs`
- maintenance integration for all layers
- per-layer reports and integration tests

Constraint preserved:
- all higher layers remain advisory/read-only

### Phase 2: Operational Hardening

Status:
- complete

Goal:
- prove that the full 5-layer stack is stable, observable, and safe enough for limited downstream use

This phase is not about adding another cognitive layer.
It is about making the existing stack measurable and promotion-ready.

### Workstream A: Cross-Layer Evaluation

Goal:
- evaluate the full stack end-to-end, not only layer by layer

Deliverables:
- unified replay suite across all 5 layers
- drift and churn dashboards per layer
- false abstraction / false causality / policy noise scenarios
- multi-topic separation tests
- multilingual and non-ASCII stability scenarios
- long-run soak tests for identity stability of beliefs, concepts, causal patterns, and policy hints

Success metrics:
- stable per-layer counts on repeated replay
- bounded churn and drift on unchanged datasets
- no cross-topic bleed in concept, causal, or policy outputs
- stable provenance graphs across repeated runs

Failure modes:
- one layer remains stable in isolation but causes drift in upper layers
- policy noise grows even when causal counts are stable
- identity churn causes concept/causal/policy explosions

Kill switch:
- disable upper derived layers independently and continue running lower stack only

### Workstream B: Observability Hardening

Goal:
- make live cognitive state inspectable without reading raw cache files

Deliverables:
- per-layer counts exposed through maintenance reports
- per-layer timing metrics
- per-layer churn / stability / suppression ratios
- candidate -> stable conversion metrics
- provenance inspection helpers
- optional debug endpoint or inspection API surface for derived layers

Success metrics:
- every maintenance cycle shows enough information to debug layer behavior
- no phase behaves as a black box under replay or soak tests

Failure modes:
- layer behavior can only be inferred indirectly from side effects
- derived-state explosions are detected too late

Kill switch:
- keep writing caches but disable expensive inspection output if latency regresses

### Workstream C: Safe Downstream Experiment

Goal:
- choose exactly one narrow downstream use case and validate it under explicit gates

Selected experiment:
- surface `policy` hints as inspection/report output only

Scope:
- hints visible in reports or debug API only
- no recall reranking
- no automatic actions
- no hidden control loop

Why this experiment:
- user-visible value
- low blast radius
- strong provenance already exists
- easier to evaluate than concept-aware compression or policy-guided execution

Success metrics:
- surfaced hints remain traceable and understandable
- hint volume stays bounded
- no degradation in recall behavior
- no misleading high-confidence hints on regression scenarios

Failure modes:
- recommendation explosion
- fake authority from weak causal seeds
- unclear or low-signal recommendations

Kill switch:
- hide policy output from public/report surfaces while keeping internal generation for evaluation

### Phase 3: Promotion Review & Controlled Promotion

Status:
- complete

Delivered:
- Candidate A (policy surfaced output): promoted as stable public API
- Candidate B (belief-aware recall): shadow scoring, evidence collection (53 queries)
- Candidate C (concept-assisted grouping): deferred
- Shadow evidence: 100% top-k overlap, 45% movement, 29μs latency
- Decision: A promoted, B limited influence, C deferred

See [PHASE_3_PROMOTION_PLAN.md](PHASE_3_PROMOTION_PLAN.md).

### Phase 4: Limited Influence Activation

Status:
- complete

Delivered:
- Tri-state mode: Off / Shadow / Limited (BeliefRerankMode enum, AtomicU8)
- Bounded reranking: ±5% score cap, ±2 positional shift cap
- Scope guards: min 4 results, top_k ≤ 20, belief_coverage > 0
- LimitedRerankReport with comprehensive diagnostic metrics
- recall_raw() / recall_finalize() / recall_core() refactor (fixes double-rerank bug)
- Evidence pass: 82 queries, 42.7% BETTER, 0% WORSE, all safety gates passed
- Decision: wider rollout approved (default-on for eval/internal, guardrails retained)

See [PHASE_4_PLAN.md](PHASE_4_PLAN.md).

## Readiness Gates v2

These gates must be passed before any advisory layer can influence recall or behavior.

### Gate A: Belief -> Recall-Aware Ranking — **PASSED (Phase 4)**

Status: **wider rollout approved** (2026-03-11)

Evidence:
- 82-query dual-mode evaluation: 42.7% BETTER, 0% WORSE
- avg top-k overlap 0.945, positional shift bounded ±2
- contradiction leakage: 0% worsened
- latency delta: -26μs (no regression)
- guardrails retained: ±5% score cap, ±2 pos cap, scope guards

Original requirements (all met):
- grouping benchmark remains above target
- replay churn remains within target
- unresolved belief rate stays bounded on mixed datasets
- recall benchmark improves or stays flat

### Gate B: Concept -> Compression or Ranking Influence — **KEEP SHADOW**

Status: **keep shadow** — shadow evaluation completed (2026-03-11)

Shadow evaluation results — dual verdict (80 runs: 60 synthetic + 20 practical):
- **Verdict A (concept layer health): HEALTHY** — concept.rs works correctly under sufficient belief density
  - Multi-concept profile: 100% formation, 49.4% coverage, 92% USEFUL, 0 false merges
- **Verdict B (practical viability): BLOCKED** — upstream belief fragmentation prevents concept formation
  - Practical: 0/20 form concepts — tag diversity → singleton beliefs → 0 seeds per partition
  - This is a belief pipeline constraint, not a concept engine bug
- Safety: 0% misleading, 0 false merges, 0 recall degradation, deterministic identity
- See: CONCEPT_SHADOW_EVAL_REPORT.md, CONCEPT_CLUSTERING_RETUNE_REPORT.md

Upstream bottleneck identified:
- Tag-based belief grouping produces unique keys on diverse corpora
- Most records get singleton beliefs that never reach stability >= 2.0
- Concept formation requires >= 2 seeds in same (namespace, semantic_type) partition
- This is a belief pipeline constraint, not a concept engine bug

Requirements:
- [x] concept identity stability across replay (measured: stable across 25 cycles)
- [x] low false abstraction rate (measured: 0% misleading, 0 false merges in 80 runs)
- [x] no regression in record-level recall fidelity (measured: 0 degradation in 80 runs)
- [ ] coverage ≥ 30% on practical query sets (blocked by belief density — currently 0%)

Path forward (out of current scope):
- Natural corpus growth may resolve (more records with overlapping tags)
- Belief subclustering enhancement could merge related-but-not-identical tag sets
- Do NOT force by relaxing safety thresholds

Do not promote if:
- concept collapse hides important distinctions
- concept count or membership drifts on unchanged streams

### Gate C: Causal -> Decision Weighting

Require:
- confounder tests remain green
- causal precision benchmark passes agreed target
- temporal-only patterns remain weak unless reinforced

Do not promote if:
- false causal rules dominate output
- causal advice contradicts explicit provenance

### Gate D: Policy -> Surfaced Recommendations

Require:
- policy hints derive only from stabilized lower layers
- suppression rules behave predictably
- recommendation volume and confidence stay bounded
- every hint has full provenance

Do not promote if:
- hints cannot be explained
- weak evidence generates strong recommendations

### Gate E: Policy -> Behavioral Influence

Status:
- explicitly out of scope for current roadmap stage

Require in the future:
- separate approval
- adversarial evaluation
- rollbackable influence path
- explicit human-visible execution boundary

## Unified Evaluation Plan

The evaluation plan for Phase 2 should be cross-layer, not per-module only.

### 1. Replay Pack

Scenarios:
- stable preferences with paraphrases
- contextual preferences with non-conflicting variants
- conflicting decisions over time
- deploy/safety incident chains
- multi-topic mixed namespace datasets
- multilingual/non-ASCII datasets

Measure:
- record support/conflict stability
- belief churn and unresolved ratio
- concept candidate count and identity stability
- causal candidate count, precision, and drift
- policy hint count, suppression ratio, and confidence spread

### 2. Soak Pack

Scenarios:
- 20-100 maintenance cycles on fixed datasets
- mixed causal and non-causal streams
- gradual data growth without semantic change

Measure:
- candidate growth per layer
- identity churn per layer
- average score variance per layer
- maintenance latency and per-phase timing

### 3. Adversarial Pack

Scenarios:
- confounders
- shared tags but unrelated topics
- contradictory records with narrow scope
- temporal adjacency without causal link
- strong language but weak evidence

Measure:
- false belief merges
- false abstractions
- false causal promotions
- policy overreach

## Observability Plan

Minimum observability expected after Phase 2:

- per-phase timing
- per-layer total counts
- per-layer stable/candidate/rejected counts where applicable
- belief churn rate
- concept abstraction score distribution
- causal strength distribution
- policy strength distribution
- suppression and unresolved ratios
- cache freshness / startup-empty semantics for derived layers

Recommended next implementation steps:

1. add per-phase timing into `MaintenanceReport`
2. add per-layer identity stability counters
3. add candidate -> stable conversion counters
4. add lightweight inspection helpers for derived artifacts

## Selected Safe Downstream Experiment

Experiment:
- expose policy hints as surfaced inspection output only

Implementation shape:
- add policy hints to an inspection/debug API or reporting surface
- keep them off recall and execution paths
- require provenance in every surfaced hint
- include hint state, strength, and supporting IDs

Why this first:
- smallest blast radius
- easiest to explain and review
- strongest immediate user-facing value without behavior risk

Explicitly not selected yet:
- ~~belief-aware recall ranking~~ → **promoted (Phase 4, wider rollout)**
- concept-aware compression
- causal-weighted planning
- policy-guided automatic action selection

## Risk Register

### R1. Recall Regression

Risk:
- derived layers accidentally alter ranking before they are validated

Impact:
- medium to high

Likelihood:
- medium

Mitigation:
- keep advisory-only boundary
- benchmark recall latency and result quality before every promotion
- keep feature flags around any future ranking changes

Stop condition:
- structured recall regressions exceed agreed benchmark threshold

### R2. Maintenance Complexity Explosion

Risk:
- more phases make maintenance slower, harder to reason about, and harder to debug

Impact:
- high

Likelihood:
- high

Mitigation:
- keep each layer independently disableable
- measure per-phase timing
- require bounded runtime per phase
- avoid cross-phase side effects

Stop condition:
- maintenance latency or memory overhead regresses beyond target envelope

### R3. Serialization and Backward Compatibility Failure

Risk:
- new stores or record fields break old brains or cross-version compatibility

Impact:
- high

Likelihood:
- medium

Mitigation:
- additive schema only where possible
- separate stores for `belief`, `concept`, `causal`, `policy`
- migration tests on old cognitive datasets

Stop condition:
- any failure to open existing user brains without manual repair

### R4. False Belief Revision

Risk:
- noisy records incorrectly flip the current winner

Impact:
- high

Likelihood:
- medium

Mitigation:
- revision thresholds
- unresolved state
- hysteresis and stability tracking
- curated contradiction tests

Stop condition:
- belief winners flip repeatedly on semantically stable replay sets

### R5. False Abstraction

Risk:
- dynamic concepts merge patterns that should stay separate

Impact:
- high

Likelihood:
- medium

Mitigation:
- concepts remain inspection-first
- require stable support across cycles before any future promotion
- preserve links back to source records

Stop condition:
- concept-derived behavior hides important distinctions in regression tests

### R6. Spurious Causality

Risk:
- the system learns attractive but false causal rules from temporal correlation

Impact:
- very high

Likelihood:
- high

Mitigation:
- conservative thresholds
- synthetic confounder tests
- repeated outcome consistency
- keep causal advisory-only

Stop condition:
- causal rules fail precision tests or generate contradictory downstream hints

### R7. Unsafe Policy Influence

Risk:
- incorrect cognition changes behavior in ways users do not expect

Impact:
- very high

Likelihood:
- medium

Mitigation:
- advisory-only first
- explicit provenance on every policy hint
- never auto-execute from policy layer

Stop condition:
- any recommendation cannot be traced back to supporting state

## Negative Utility Triggers

The project is at risk of negative utility if any of the following happens:

- new layers add internal complexity without measurable quality gains
- recall gets slower while answer quality does not improve
- beliefs become unstable across identical replays
- concepts compress away edge cases users actually care about
- causal rules encode correlations as laws
- policy hints become trusted before their upstream layers are validated
- observability is too weak to explain why a layer emitted what it emitted

## Safe Integration Rules

- Every new layer starts in shadow mode or advisory mode.
- No derived layer influences recall ranking on first release.
- No derived layer influences agent behavior automatically on first release.
- Every derived artifact keeps provenance links to underlying records.
- Every phase exposes timing and count metrics.
- Every phase has regression, replay, and soak coverage.
- Promotion to influence mode requires an explicit gate review.

## Immediate Next Steps

1. ~~update COGNITIVE_ROADMAP.md~~ — done
2. ~~add unified cross-layer replay and soak suites~~ — done (Phase 2 Workstream A)
3. ~~add per-phase timing and stability metrics to MaintenanceReport~~ — done (Phase 2 Workstream B)
4. ~~expose policy hints through an inspection-only surface~~ — done (Phase 2 Workstream C, promoted in Phase 3)
5. ~~re-evaluate readiness before any recall or behavior influence work~~ — done (Phase 3 + Phase 4)

### Current Position (post Phase 4, Step 10 complete)

**Candidate B: STABILIZED WIDER ROLLOUT** — all rollout steps complete (2026-03-11).

Step 9 (cross-layer eval with rerank enabled):
- 6 tests, 10 queries across 4 scenarios, all safety gates PASS
- Avg top-k overlap: 1.000, max positional shift: ±1
- Cross-layer invariants fully preserved under active reranking
- 20-cycle soak: stable, no drift

Step 10 decision: **KEEP AND WIDER-ROLLOUT CONFIRMED**
- No revert grounds: zero regressions across 82 + 10 queries
- Not "keep experimental" — evidence is strong enough for stabilized wider rollout
- Not an argument for aggressive rerank — an argument for holding bounded rollout
- All guardrails retained: scope guards, ±5% score cap, ±2 pos cap, tri-state mode, rollback path

### Steady-State Monitoring Cadence

The project is now in monitoring mode, not feature development mode.

Periodic metrics to track:
1. Belief coverage trend (currently ~15%)
2. Movement rate under Limited mode
3. Contradiction leakage (target: 0%)
4. Latency drift (target: < 2ms per recall)
5. Rollback incidents (target: 0)
6. % better / same / worse on periodic query packs

### What NOT To Do

1. Do NOT promote Candidate C to recall influence until wider coverage re-evaluation passes (≥ 30% on practical queries)
2. Do NOT proceed to causal/policy behavior influence
3. Do NOT relax any guardrails without new evidence pass
4. Do NOT increase rerank multipliers or positional caps
5. Do NOT make reranking unconditional/global default without rollback
6. Do NOT add concept-aware recall ranking, concept compression, or behavior influence to Candidate C scope

## Current Position (2026-03-11)

Aura has a complete 5-layer cognitive stack with one hardened influence path:

`Record -> Belief -> [recall rerank] -> Concept -> Causal Pattern -> Policy`

Completed milestones:
- Phase 0: epistemic base stabilized
- Phase 0.5: belief formation hardened
- Phase 1: read-only cognitive stack (all 5 layers)
- Phase 2: operational hardening (cross-layer eval, observability, surfaced policy output)
- Phase 3: promotion review (A promoted, B shadow-approved, C deferred)
- Phase 4: limited influence activation and hardening (B wider rollout, evidence-backed, then synthetic-campaign validated)

What is active:
- Belief-aware recall reranking: **fully hardened wider rollout** (default-on for eval/internal, all guardrails retained)
- Policy surfaced output: **stable public API**

What is in shadow evaluation:
- Concept-assisted grouping (Candidate C) — KEEP SHADOW after full shadow evaluation (dual verdict)
  - **Verdict A: HEALTHY** — concept.rs works correctly under sufficient belief density (multi-concept: 49.4% coverage, 92% USEFUL)
  - **Verdict B: BLOCKED** — practical corpora: 0% — upstream belief density blocks formation (belief pipeline constraint, not concept engine bug)
  - Safe but not useful enough for inspection-only promotion yet

What remains deferred or blocked:
- Concept-aware recall ranking — not started, depends on Candidate C coverage gate
- Causal-weighted planning — not started
- Policy-guided behavior influence — explicitly out of scope

Current operating guidance:
- treat Candidate B as frozen baseline behavior
- keep rollback path and all guardrails intact
- do not spend more roadmap effort on Candidate B unless new evidence shows regression

503 tests passing. Zero regressions. Core value preserved: fast, local, mathematical cognition without external models.
