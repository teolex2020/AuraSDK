//! CognitiveRecord — the unified memory unit.
//!
//! Combines metadata from both aura-memory (SDR-based) and aura-cognitive
//! (hierarchical decay) into a single struct exposed via PyO3.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::levels::Level;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// A single cognitive memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
pub struct Record {
    /// Unique identifier (12-char hex).
    pub id: String,
    /// Memory content text.
    pub content: String,
    /// Cognitive hierarchy level.
    pub level: Level,
    /// Activation strength (0.0–1.0). Decays over time.
    pub strength: f32,
    /// Number of times this record has been activated (recalled).
    pub activation_count: u32,
    /// Unix timestamp of creation.
    pub created_at: f64,
    /// Unix timestamp of last activation.
    pub last_activated: f64,
    /// Classification tags.
    pub tags: Vec<String>,
    /// Bidirectional connections to other records (id → weight).
    pub connections: HashMap<String, f32>,
    /// Connection relationship types (id → type).
    /// Types: "causal", "reflective", "associative", "coactivation", or custom.
    #[serde(default)]
    pub connection_types: HashMap<String, String>,
    /// Content type: "text", "code", "json", "image_ref".
    pub content_type: String,
    /// Type-specific metadata.
    pub metadata: HashMap<String, String>,
    /// Link to aura-memory SDR engine record ID.
    pub aura_id: Option<String>,
    /// Causal parent record ID (for decision rationale chains).
    pub caused_by_id: Option<String>,
    /// Isolation namespace. Records in different namespaces are invisible to each
    /// other during recall/search unless explicitly requested.
    /// Default: "default". Empty string is NOT valid.
    #[serde(default = "default_namespace")]
    pub namespace: String,
}

/// Default namespace for records.
pub const DEFAULT_NAMESPACE: &str = "default";

fn default_namespace() -> String {
    DEFAULT_NAMESPACE.to_string()
}

impl Record {
    /// Create a new record with defaults.
    pub fn new(content: String, level: Level) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let id = Self::generate_id();

        Self {
            id,
            content,
            level,
            strength: 1.0,
            activation_count: 0,
            created_at: now,
            last_activated: now,
            tags: Vec::new(),
            connections: HashMap::new(),
            connection_types: HashMap::new(),
            content_type: "text".to_string(),
            metadata: HashMap::new(),
            aura_id: None,
            caused_by_id: None,
            namespace: DEFAULT_NAMESPACE.to_string(),
        }
    }

    /// Generate a 12-char hex ID.
    pub fn generate_id() -> String {
        uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
    }

    /// Composite importance score (0.0–1.0+).
    ///
    /// Formula: strength(40%) + level(25%) + connections(20%) + activations(15%)
    pub fn importance(&self) -> f32 {
        let level_score = self.level.value() as f32 / 4.0;
        let conn_score = (self.connections.len() as f32 / 50.0).min(1.0);
        let act_score = (self.activation_count as f32 / 20.0).min(1.0);

        0.40 * self.strength
            + 0.25 * level_score
            + 0.20 * conn_score
            + 0.15 * act_score
    }

    /// Activate this record (boost strength, update timestamp).
    pub fn activate(&mut self) {
        self.strength = (self.strength + 0.2).min(1.0);
        self.activation_count += 1;
        self.last_activated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
    }

    /// Apply daily decay based on level.
    ///
    /// Uses adaptive decay: rate interpolates from base toward 0.999
    /// as activation_count grows (ceiling effect for frequently used records).
    pub fn apply_decay(&mut self) {
        let base_rate = self.level.decay_rate();
        let ceiling_factor = (self.activation_count as f32 / 10.0).min(1.0);
        let effective_rate = base_rate + (0.999 - base_rate) * ceiling_factor;
        self.strength *= effective_rate;
    }

    /// Whether this record is still alive (not archived).
    pub fn is_alive(&self) -> bool {
        self.strength >= 0.05
    }

    /// Whether this record is eligible for promotion.
    ///
    /// Requires: activation_count >= 5, strength >= 0.7, level < IDENTITY.
    pub fn can_promote(&self) -> bool {
        self.activation_count >= 5
            && self.strength >= 0.7
            && self.level < Level::Identity
    }

    /// Promote to the next level, if eligible.
    pub fn promote(&mut self) -> bool {
        if let Some(next) = self.level.promote() {
            self.level = next;
            true
        } else {
            false
        }
    }

    /// Add a bidirectional connection to another record.
    pub fn add_connection(&mut self, other_id: &str, weight: f32) {
        let clamped = weight.clamp(0.0, 1.0);
        self.connections.insert(other_id.to_string(), clamped);
    }

    /// Add a typed bidirectional connection to another record.
    pub fn add_typed_connection(&mut self, other_id: &str, weight: f32, relationship: &str) {
        self.add_connection(other_id, weight);
        self.connection_types.insert(other_id.to_string(), relationship.to_string());
    }

    /// Get the relationship type for a connection (None if untyped).
    pub fn connection_type(&self, other_id: &str) -> Option<&str> {
        self.connection_types.get(other_id).map(|s| s.as_str())
    }

    /// Days since creation.
    pub fn age_days(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        (now - self.created_at) / 86400.0
    }

    /// Validate a namespace string.
    ///
    /// Rules: non-empty, max 64 chars, ASCII alphanumeric + hyphens + underscores.
    pub fn validate_namespace(ns: &str) -> Result<(), String> {
        if ns.is_empty() {
            return Err("Namespace cannot be empty".into());
        }
        if ns.len() > 64 {
            return Err("Namespace cannot exceed 64 characters".into());
        }
        if !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err("Namespace must contain only ASCII alphanumeric, hyphens, or underscores".into());
        }
        Ok(())
    }

    /// Days since last activation.
    pub fn days_since_activation(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        (now - self.last_activated) / 86400.0
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Record {
    #[getter]
    fn get_id(&self) -> &str { &self.id }
    #[getter]
    fn get_content(&self) -> &str { &self.content }
    #[getter]
    fn get_level(&self) -> Level { self.level }
    #[getter]
    fn get_strength(&self) -> f32 { self.strength }
    #[getter]
    fn get_activation_count(&self) -> u32 { self.activation_count }
    #[getter]
    fn get_created_at(&self) -> f64 { self.created_at }
    #[getter]
    fn get_last_activated(&self) -> f64 { self.last_activated }
    #[getter]
    fn get_tags(&self) -> Vec<String> { self.tags.clone() }
    #[getter]
    fn get_connections(&self) -> HashMap<String, f32> { self.connections.clone() }
    #[getter]
    fn get_connection_types(&self) -> HashMap<String, String> { self.connection_types.clone() }
    #[getter]
    fn get_content_type(&self) -> &str { &self.content_type }
    #[getter]
    fn get_metadata(&self) -> HashMap<String, String> { self.metadata.clone() }
    #[getter]
    fn get_aura_id(&self) -> Option<String> { self.aura_id.clone() }
    #[getter]
    fn get_caused_by_id(&self) -> Option<String> { self.caused_by_id.clone() }
    #[getter]
    fn get_namespace(&self) -> &str { &self.namespace }
    #[getter]
    fn get_importance(&self) -> f32 { self.importance() }

    fn __repr__(&self) -> String {
        let ns_suffix = if self.namespace == DEFAULT_NAMESPACE {
            String::new()
        } else {
            format!(", ns='{}'", self.namespace)
        };
        format!(
            "Record(id='{}', level={}, strength={:.2}{}, content='{}...')",
            self.id,
            self.level.name(),
            self.strength,
            ns_suffix,
            &self.content.chars().take(40).collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_record() {
        let rec = Record::new("Hello world".into(), Level::Working);
        assert_eq!(rec.content, "Hello world");
        assert_eq!(rec.level, Level::Working);
        assert_eq!(rec.strength, 1.0);
        assert_eq!(rec.activation_count, 0);
        assert!(rec.is_alive());
        assert!(!rec.can_promote());
    }

    #[test]
    fn test_activate() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.strength = 0.5;
        rec.activate();
        assert_eq!(rec.strength, 0.7);
        assert_eq!(rec.activation_count, 1);
    }

    #[test]
    fn test_decay() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.apply_decay();
        // With 0 activations, rate = 0.80
        assert!((rec.strength - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_decay() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.activation_count = 10;
        rec.apply_decay();
        // With 10 activations, rate → 0.999
        assert!((rec.strength - 0.999).abs() < 0.01);
    }

    #[test]
    fn test_promotion() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.activation_count = 5;
        rec.strength = 0.8;
        assert!(rec.can_promote());
        assert!(rec.promote());
        assert_eq!(rec.level, Level::Decisions);
    }

    #[test]
    fn test_importance() {
        let rec = Record::new("test".into(), Level::Identity);
        // strength=1.0 (0.4) + level=4/4 (0.25) + conn=0 (0) + act=0 (0) = 0.65
        assert!((rec.importance() - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_is_alive() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.strength = 0.05;
        assert!(rec.is_alive());
        rec.strength = 0.04;
        assert!(!rec.is_alive());
    }

    #[test]
    fn test_typed_connection() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.add_typed_connection("other-1", 0.8, "causal");
        rec.add_typed_connection("other-2", 0.5, "reflective");
        rec.add_connection("other-3", 0.3); // untyped

        assert_eq!(rec.connections.len(), 3);
        assert_eq!(rec.connection_types.len(), 2);
        assert_eq!(rec.connection_type("other-1"), Some("causal"));
        assert_eq!(rec.connection_type("other-2"), Some("reflective"));
        assert_eq!(rec.connection_type("other-3"), None);
    }

    #[test]
    fn test_typed_connection_serde() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.add_typed_connection("x", 0.7, "associative");

        let json = serde_json::to_string(&rec).unwrap();
        let restored: Record = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.connection_type("x"), Some("associative"));
        assert_eq!(restored.connections.get("x").copied(), Some(0.7));
    }

    #[test]
    fn test_backward_compat_no_types() {
        // Old records without connection_types should deserialize fine
        // Serialize a record, strip connection_types, and re-deserialize
        let mut rec = Record::new("old record".into(), Level::Working);
        rec.add_connection("other", 0.5);

        let mut json_val: serde_json::Value = serde_json::to_value(&rec).unwrap();
        // Remove connection_types to simulate old data format
        json_val.as_object_mut().unwrap().remove("connection_types");

        let restored: Record = serde_json::from_value(json_val).unwrap();
        assert_eq!(restored.connections.len(), 1);
        assert!(restored.connection_types.is_empty()); // #[serde(default)] ensures this
        assert_eq!(restored.connection_type("other"), None);
    }

    // ── Namespace tests ───────────────────────────────────────────

    #[test]
    fn test_default_namespace() {
        let rec = Record::new("test".into(), Level::Working);
        assert_eq!(rec.namespace, "default");
    }

    #[test]
    fn test_custom_namespace() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.namespace = "project-x".to_string();
        assert_eq!(rec.namespace, "project-x");
    }

    #[test]
    fn test_backward_compat_no_namespace() {
        // Old records without namespace field should deserialize with "default"
        let rec = Record::new("old record".into(), Level::Working);
        let mut json_val: serde_json::Value = serde_json::to_value(&rec).unwrap();
        json_val.as_object_mut().unwrap().remove("namespace");
        let restored: Record = serde_json::from_value(json_val).unwrap();
        assert_eq!(restored.namespace, "default");
    }

    #[test]
    fn test_namespace_serialization_roundtrip() {
        let mut rec = Record::new("test".into(), Level::Working);
        rec.namespace = "custom-ns".to_string();
        let json = serde_json::to_string(&rec).unwrap();
        let restored: Record = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.namespace, "custom-ns");
    }

    #[test]
    fn test_validate_namespace() {
        assert!(Record::validate_namespace("default").is_ok());
        assert!(Record::validate_namespace("project-x").is_ok());
        assert!(Record::validate_namespace("test_ns").is_ok());
        assert!(Record::validate_namespace("ns123").is_ok());
        assert!(Record::validate_namespace("").is_err());
        assert!(Record::validate_namespace("ab cd").is_err());
        assert!(Record::validate_namespace("ns/path").is_err());
        assert!(Record::validate_namespace(&"a".repeat(65)).is_err());
        assert!(Record::validate_namespace(&"a".repeat(64)).is_ok());
    }
}
