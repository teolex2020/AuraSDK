# V3 Operator Workflows

This file collects the most useful bounded operator workflows added across v2 and v3:

- correction and correction-log inspection
- record and recall explainability
- cross-namespace analytics
- policy and belief-instability operator inspection

These examples are read-only unless a correction API is explicitly called.

---

## 1. Correction Workflow

### Python

```python
from aura import Aura

brain = Aura("./data")

# Apply a targeted correction with a persistent reason trail.
brain.invalidate_causal_pattern_with_reason(
    "causal-prod-deploy-1",
    "superseded_by_new_rollout_evidence",
)

# Inspect the persisted correction log.
entries = brain.get_correction_log()
print(entries[-1]["target_kind"], entries[-1]["reason"])
```

### HTTP

```text
GET /correction-log?target_kind=causal_pattern&target_id=causal-prod-deploy-1&limit=20
```

### MCP

```json
{
  "tool": "correction_log",
  "arguments": {
    "target_kind": "causal_pattern",
    "target_id": "causal-prod-deploy-1",
    "limit": 20
  }
}
```

---

## 2. Record Explainability

### Python

```python
from aura import Aura

brain = Aura("./data")

item = brain.explain_record("rec_123")
print(item["belief"])
print(item["causal_patterns"])
print(item["policy_hints"])
```

### HTTP

```text
GET /explain-record?record_id=rec_123
```

### MCP

```json
{
  "tool": "explain_record",
  "arguments": {
    "record_id": "rec_123"
  }
}
```

---

## 3. Recall Explainability

Use this when you need to understand why the current recall stack ranked records the way it did.

### Python

```python
from aura import Aura

brain = Aura("./data")
brain.enable_full_cognitive_stack()

explanation = brain.explain_recall(
    "deploy rollback stability",
    top_k=5,
)

for item in explanation["items"]:
    print(item["record_id"], item["trace"]["final_score"])
```

### HTTP

```text
GET /explain-recall?query=deploy%20rollback%20stability&top_k=5
```

### MCP

```json
{
  "tool": "explain_recall",
  "arguments": {
    "query": "deploy rollback stability",
    "top_k": 5
  }
}
```

---

## 4. Explainability Bundle

Use this when a UI or operator tool wants one bounded inspect object instead of stitching multiple APIs together.

### Python

```python
bundle = brain.explainability_bundle("rec_123")
print(bundle["provenance"]["narrative"])
print(bundle["belief_instability"])
print(bundle["maintenance_trends"])
```

### HTTP

```text
GET /explainability-bundle?record_id=rec_123
```

### MCP

```json
{
  "tool": "explainability_bundle",
  "arguments": {
    "record_id": "rec_123"
  }
}
```

---

## 5. Cross-Namespace Analytics

Use this for dashboard-style inspection across isolated namespaces. It does not affect recall.

### Python

```python
digest = brain.cross_namespace_digest(
    namespaces=["default", "sandbox"],
    top_concepts_limit=3,
    min_record_count=2,
    include_dimensions=["concepts", "tags", "causal", "belief_states", "corrections"],
)

print(digest["namespaces"][0]["belief_state_summary"])
print(digest["pairs"][0]["tag_jaccard"])
```

### HTTP

```text
GET /cross-namespace-digest?namespaces=default,sandbox&top_concepts_limit=3&min_record_count=2&include_dimensions=concepts,tags,causal,belief_states,corrections
```

### MCP

```json
{
  "tool": "cross_namespace_digest",
  "arguments": {
    "namespaces": ["default", "sandbox"],
    "top_concepts_limit": 3,
    "min_record_count": 2,
    "include_dimensions": ["concepts", "tags", "causal", "belief_states", "corrections"]
  }
}
```

---

## 6. Operator Inspection

### Policy Lifecycle

```json
{
  "tool": "policy_lifecycle",
  "arguments": {
    "namespace": "default",
    "limit": 10,
    "action_limit": 8,
    "domain_limit": 12
  }
}
```

### Belief Instability

```json
{
  "tool": "belief_instability",
  "arguments": {
    "min_volatility": 0.2,
    "max_stability": 1.0,
    "limit": 10
  }
}
```

### Rust Grouped API

```rust
use aura::Aura;

let brain = Aura::open("./data")?;

let summary = brain.operator_api().get_policy_lifecycle_summary(Some(8), Some(12));
let pressure = brain.operator_api().get_policy_pressure_report(Some("default"), Some(10));
let instability = brain.operator_api().get_belief_instability_summary();
```
