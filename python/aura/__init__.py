"""Aura — Cognitive Memory for AI Agents.

Pure Rust implementation with Python bindings.
No embeddings required. No vendor lock-in.
"""

from aura._core import (
    # Main
    Aura,
    Level,
    Record,

    # Tag & Trust Configuration
    TagTaxonomy,
    TrustConfig,

    # Living Memory (Background Maintenance)
    MaintenanceConfig,
    MaintenanceReport,
    ArchivalRule,
    DecayReport,
    ReflectReport,
    ConsolidationReport,

    # Identity
    AgentPersona,
    PersonaTraits,

    # Circuit Breaker
    CircuitBreakerConfig,
)

from aura.events import AuraEvents

__version__ = "1.5.2"
__all__ = [
    "Aura",
    "AuraEvents",
    "Level",
    "Record",
    "TagTaxonomy",
    "TrustConfig",
    "MaintenanceConfig",
    "MaintenanceReport",
    "ArchivalRule",
    "DecayReport",
    "ReflectReport",
    "ConsolidationReport",
    "AgentPersona",
    "PersonaTraits",
    "CircuitBreakerConfig",
]
