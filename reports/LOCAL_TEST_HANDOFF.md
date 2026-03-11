# Local Test Handoff

## Build Artifact

Current local wheel built from the latest AuraSDK state:

- `D:\AuraSDK-verify\target\wheels\aura_memory-1.4.0-cp312-cp312-win_amd64.whl`

Build command used:

```powershell
maturin build --release
```

## Install Command For `health-secretary`

From the target project environment:

```powershell
python -m pip install --force-reinstall D:\AuraSDK-verify\target\wheels\aura_memory-1.4.0-cp312-cp312-win_amd64.whl
```

## Important Note

The package version is still `1.4.0`, but the wheel contains the new post-1.4.0 cognitive changes from the current local branch. Treat this as a local validation build, not a public release.

## What Changed In This Build

### Stable and Product-Ready

- `A`: surfaced policy output is stable
- `B`: belief-aware recall reranking is fully hardened
- `C`: concept grouping is now promoted in inspection-only mode

### Major Additions

- richer epistemic record state:
  - `confidence`
  - `support_mass`
  - `conflict_mass`
  - `volatility`
- belief layer:
  - claim grouping
  - resolved/unresolved/singleton state
  - bounded belief-aware reranking
- concept layer:
  - canonical-feature similarity path
  - surfaced concept output
  - deterministic concept identity
- causal layer:
  - advisory causal candidates
- policy layer:
  - surfaced policy hints
- expanded maintenance/inspection observability:
  - timings
  - stability counters
  - inspection helpers

## New Public Inspection Surfaces Worth Testing

### Policy

- `get_surfaced_policy_hints(limit)`
- `get_surfaced_policy_hints_for_namespace(namespace, limit)`

### Concept

- `get_surfaced_concepts(limit)`
- `get_surfaced_concepts_for_namespace(namespace, limit)`

### Belief / Internal Cognitive State

- `get_beliefs(state_filter)`
- `get_concepts(state_filter)`
- `get_causal_patterns(state_filter)`
- `get_policy_hints(state_filter)`

## Practical Test Focus For `health-secretary`

### 1. Baseline compatibility

Verify existing 1.4.0 flows still work:

- store
- recall
- recall_structured
- maintenance
- snapshots / export if used

### 2. Belief-aware recall behavior

Look for:

- no regressions in result quality
- slight improvement where repeated stable beliefs exist
- no chaotic reranking

### 3. Surfaced concepts

Check whether concept surfaces become useful in real app memory:

- are surfaced concepts non-empty?
- are they topic-correct?
- do they help inspection/debugging?

### 4. Surfaced policy hints

Check whether policy hints are:

- bounded
- understandable
- provenance-backed
- not noisy

### 5. Zero hidden behavior regressions

Confirm the new cognitive layers do not:

- mutate recall unexpectedly
- create instability across repeated runs
- break latency expectations in the app

## Suggested Smoke Test Sequence

1. Install the local wheel in the `health-secretary` environment.
2. Run the app's existing memory-dependent flows unchanged.
3. Compare current recall behavior against the old installed 1.4.0 behavior.
4. Inspect surfaced concepts and surfaced policy hints on realistic app data.
5. Run at least one maintenance cycle and confirm nothing degrades.

## Expected Outcome

This build should behave like a stronger 1.4.0-compatible local cognitive layer:

- old baseline behavior should still work
- belief-aware rerank may improve some recall cases
- surfaced concepts and policy hints should now be available for practical inspection

## If Something Breaks

Primary rollback:

```powershell
python -m pip install --force-reinstall aura-memory==1.4.0
```

Or reinstall the previously known-good internal wheel if one exists in the target environment.
