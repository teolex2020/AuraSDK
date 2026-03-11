# Aura v2 Positioning

## One-Line Positioning

Aura v2 is a local cognitive memory substrate for AI agents: fast memory, belief-aware recall, surfaced concepts, causal inspection, and policy hints without LLM calls or cloud infrastructure.

## What Changed In v2

Aura is no longer only "memory that stores and recalls." It now provides a bounded cognitive stack:

`Record -> Belief -> Concept -> Causal Pattern -> Policy`

The important distinction is that Aura still stays:

- local
- deterministic
- lightweight
- mathematically driven
- advisory-first

It does not become an orchestration framework or a cloud reasoning service.

## Core Product Promise

Aura gives agents a memory substrate that becomes more structured over time:

- remembers records
- resolves beliefs
- surfaces reusable concepts
- detects causal candidates
- emits policy hints

All of that happens offline, with bounded latency and without external models.

## Who It Is For

- AI agent developers who want persistent cognition without vector DB + LLM glue
- local-first products that cannot depend on cloud memory services
- teams that need explainable memory behavior and provenance
- builders who want an embeddable Rust core rather than a hosted black box

## What Makes Aura Different

### Not a vector-memory wrapper

Aura is not just "store embeddings, query embeddings."

It models:

- decay
- promotion
- epistemic state
- belief conflict
- concept emergence
- causal patterns
- policy hints

### Not an LLM memory layer

Aura does not require an LLM call to maintain memory quality.

### Not an agent framework

Aura is the cognitive substrate under an agent, not the orchestration shell around it.

## Product Story

### Aura v1.4.x

- fast local memory
- decay/promotion
- semantic memory roles
- strong storage/recall baseline

### Aura v2

- belief-aware reranking promoted
- surfaced policy hints promoted
- surfaced concept grouping promoted in inspection-only mode
- causal layer available for inspection/advisory use

## Current Product Boundaries

Aura v2 intentionally stops short of:

- autonomous behavior control
- policy-driven execution
- concept-based compression
- hidden action selection
- mandatory hosted service dependency

This is a design choice, not a missing implementation.

## Recommended External Positioning

Use wording like:

"Aura is a cognitive memory engine for AI agents: local memory, belief-aware recall, surfaced concepts, causal inspection, and policy hints in a lightweight Rust core."

Avoid wording like:

- "AGI memory system"
- "fully autonomous reasoning engine"
- "self-improving general intelligence"

## Commercial Positioning

The open product should be visible enough to drive adoption.
The differentiating internals should remain monetizable.

Recommended positioning:

- `Aura Community`: open SDK and core local memory experience
- `Aura Pro`: advanced cognitive layers, hardened evaluation, enterprise support, commercial licensing

