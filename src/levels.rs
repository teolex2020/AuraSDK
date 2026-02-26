//! Cognitive memory hierarchy levels.
//!
//! Four-tier system with decay rates inspired by human memory:
//! - WORKING: temporary thoughts (~3 days)
//! - DECISIONS: moderate persistence (~7 days)
//! - DOMAIN: important knowledge (~14 days)
//! - IDENTITY: core identity (~70+ days)

use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Cognitive memory level — determines decay rate and importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[repr(u8)]
#[derive(Default)]
pub enum Level {
    /// Short-term working memory. Decays rapidly (0.80/day). ~3 days lifespan.
    #[default]
    Working = 1,
    /// Decision records. Moderate decay (0.90/day). ~7 days half-life.
    Decisions = 2,
    /// Domain knowledge. Slow decay (0.95/day). ~14 days half-life.
    Domain = 3,
    /// Core identity. Near-permanent (0.99/day). ~70 days half-life.
    Identity = 4,
}

impl Level {
    /// Daily decay rate for this level.
    pub fn decay_rate(&self) -> f32 {
        match self {
            Level::Working => 0.80,
            Level::Decisions => 0.90,
            Level::Domain => 0.95,
            Level::Identity => 0.99,
        }
    }

    /// Map to aura-memory DNA classification.
    pub fn to_dna(&self) -> &'static str {
        match self {
            Level::Working => "general",
            Level::Decisions => "general",
            Level::Domain => "user_core",
            Level::Identity => "user_core",
        }
    }

    /// Whether this level maps to SDR identity (protected bit range).
    pub fn is_identity_sdr(&self) -> bool {
        matches!(self, Level::Domain | Level::Identity)
    }

    /// Get the next higher level, if any.
    pub fn promote(&self) -> Option<Level> {
        match self {
            Level::Working => Some(Level::Decisions),
            Level::Decisions => Some(Level::Domain),
            Level::Domain => Some(Level::Identity),
            Level::Identity => None,
        }
    }

    /// Numeric value (1-4).
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Create from numeric value.
    pub fn from_value(v: u8) -> Option<Level> {
        match v {
            1 => Some(Level::Working),
            2 => Some(Level::Decisions),
            3 => Some(Level::Domain),
            4 => Some(Level::Identity),
            _ => None,
        }
    }

    /// Display name for the level.
    pub fn name(&self) -> &'static str {
        match self {
            Level::Working => "WORKING",
            Level::Decisions => "DECISIONS",
            Level::Domain => "DOMAIN",
            Level::Identity => "IDENTITY",
        }
    }

    /// Memory tier: "cognitive" (Working/Decisions) or "core" (Domain/Identity).
    ///
    /// Two-tier logical separation:
    /// - **Cognitive**: ephemeral working memory — session notes, recent decisions. Fast decay.
    /// - **Core**: permanent knowledge base — facts, profile, domain expertise. Slow decay.
    pub fn tier(&self) -> &'static str {
        match self {
            Level::Working | Level::Decisions => "cognitive",
            Level::Domain | Level::Identity => "core",
        }
    }

    /// Check if this level belongs to the cognitive tier (Working + Decisions).
    pub fn is_cognitive(&self) -> bool {
        matches!(self, Level::Working | Level::Decisions)
    }

    /// Check if this level belongs to the core tier (Domain + Identity).
    pub fn is_core(&self) -> bool {
        matches!(self, Level::Domain | Level::Identity)
    }
}


impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Level {
    /// Get the decay rate for this level.
    #[getter]
    fn get_decay_rate(&self) -> f32 {
        self.decay_rate()
    }

    /// Get the DNA classification string.
    #[getter]
    fn get_dna(&self) -> &str {
        self.to_dna()
    }

    /// Get the memory tier: "cognitive" or "core".
    #[getter]
    fn get_tier(&self) -> &str {
        self.tier()
    }

    /// Check if this level is in the cognitive tier.
    #[getter]
    fn get_is_cognitive(&self) -> bool {
        self.is_cognitive()
    }

    /// Check if this level is in the core tier.
    #[getter]
    fn get_is_core(&self) -> bool {
        self.is_core()
    }

    fn __repr__(&self) -> String {
        format!("Level.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_rates() {
        assert!(Level::Working.decay_rate() < Level::Decisions.decay_rate());
        assert!(Level::Decisions.decay_rate() < Level::Domain.decay_rate());
        assert!(Level::Domain.decay_rate() < Level::Identity.decay_rate());
    }

    #[test]
    fn test_promotion() {
        assert_eq!(Level::Working.promote(), Some(Level::Decisions));
        assert_eq!(Level::Decisions.promote(), Some(Level::Domain));
        assert_eq!(Level::Domain.promote(), Some(Level::Identity));
        assert_eq!(Level::Identity.promote(), None);
    }

    #[test]
    fn test_from_value() {
        assert_eq!(Level::from_value(1), Some(Level::Working));
        assert_eq!(Level::from_value(4), Some(Level::Identity));
        assert_eq!(Level::from_value(0), None);
        assert_eq!(Level::from_value(5), None);
    }

    #[test]
    fn test_ordering() {
        assert!(Level::Working < Level::Decisions);
        assert!(Level::Decisions < Level::Domain);
        assert!(Level::Domain < Level::Identity);
    }

    #[test]
    fn test_tier() {
        assert_eq!(Level::Working.tier(), "cognitive");
        assert_eq!(Level::Decisions.tier(), "cognitive");
        assert_eq!(Level::Domain.tier(), "core");
        assert_eq!(Level::Identity.tier(), "core");
    }

    #[test]
    fn test_is_cognitive_core() {
        assert!(Level::Working.is_cognitive());
        assert!(Level::Decisions.is_cognitive());
        assert!(!Level::Domain.is_cognitive());
        assert!(!Level::Identity.is_cognitive());

        assert!(!Level::Working.is_core());
        assert!(!Level::Decisions.is_core());
        assert!(Level::Domain.is_core());
        assert!(Level::Identity.is_core());
    }
}
