//! Retention policies for cognitive memory.
//!
//! Enforces age, strength, and count caps.

/// Retention policy configuration.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age in days (0 = unlimited).
    pub max_age_days: f64,
    /// Auto-delete records below this strength (0 = disabled).
    pub auto_delete_below_strength: f32,
    /// Maximum number of records (0 = unlimited).
    pub max_records: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age_days: 0.0,
            auto_delete_below_strength: 0.0,
            max_records: 0,
        }
    }
}

impl RetentionPolicy {
    /// Check if any policy is active.
    pub fn is_active(&self) -> bool {
        self.max_age_days > 0.0
            || self.auto_delete_below_strength > 0.0
            || self.max_records > 0
    }
}
