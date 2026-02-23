//! Automated maintenance scheduler.
//!
//! Runs periodic decay, consolidation, and reflection cycles.

use std::time::{Duration, Instant};

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Decay interval.
    pub decay_interval: Duration,
    /// Consolidation interval.
    pub consolidation_interval: Duration,
    /// Reflection interval.
    pub reflection_interval: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            decay_interval: Duration::from_secs(24 * 3600), // Daily
            consolidation_interval: Duration::from_secs(6 * 3600), // Every 6 hours
            reflection_interval: Duration::from_secs(12 * 3600), // Every 12 hours
        }
    }
}

/// Tracks when maintenance tasks were last run.
pub struct CognitiveScheduler {
    pub config: SchedulerConfig,
    pub last_decay: Option<Instant>,
    pub last_consolidation: Option<Instant>,
    pub last_reflection: Option<Instant>,
}

impl CognitiveScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            last_decay: None,
            last_consolidation: None,
            last_reflection: None,
        }
    }

    /// Check if decay is due.
    pub fn should_decay(&self) -> bool {
        self.last_decay
            .map(|t| t.elapsed() >= self.config.decay_interval)
            .unwrap_or(true)
    }

    /// Check if consolidation is due.
    pub fn should_consolidate(&self) -> bool {
        self.last_consolidation
            .map(|t| t.elapsed() >= self.config.consolidation_interval)
            .unwrap_or(true)
    }

    /// Check if reflection is due.
    pub fn should_reflect(&self) -> bool {
        self.last_reflection
            .map(|t| t.elapsed() >= self.config.reflection_interval)
            .unwrap_or(true)
    }

    /// Mark decay as completed.
    pub fn mark_decay(&mut self) {
        self.last_decay = Some(Instant::now());
    }

    /// Mark consolidation as completed.
    pub fn mark_consolidation(&mut self) {
        self.last_consolidation = Some(Instant::now());
    }

    /// Mark reflection as completed.
    pub fn mark_reflection(&mut self) {
        self.last_reflection = Some(Instant::now());
    }
}

impl Default for CognitiveScheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}
