//! Per-tool circuit breaker and health tracking.
//!
//! Rewritten from brain_tools.py ToolHealth class.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Per-tool circuit breaker configuration.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before circuit opens.
    pub failure_threshold: usize,
    /// Window in seconds for counting failures.
    pub failure_window_secs: u64,
    /// Cooldown in seconds after circuit opens.
    pub recovery_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            failure_window_secs: 600, // 10 min
            recovery_secs: 600,       // 10 min
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl CircuitBreakerConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

/// Per-tool health state.
struct ToolState {
    /// Recent failure timestamps.
    failures: Vec<Instant>,
    /// Circuit open until this time (None = closed).
    circuit_open_until: Option<Instant>,
}

/// Circuit breaker tracking per-tool failures.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    tools: Mutex<HashMap<String, ToolState>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            tools: Mutex::new(HashMap::new()),
        }
    }

    /// Record a tool failure. Opens circuit if threshold reached.
    pub fn record_failure(&self, tool_name: &str) {
        let now = Instant::now();
        let window = Duration::from_secs(self.config.failure_window_secs);
        let mut tools = self.tools.lock();

        let state = tools
            .entry(tool_name.to_string())
            .or_insert_with(|| ToolState {
                failures: Vec::new(),
                circuit_open_until: None,
            });

        // Keep only recent failures
        state.failures.retain(|t| now.duration_since(*t) < window);
        state.failures.push(now);

        if state.failures.len() >= self.config.failure_threshold {
            state.circuit_open_until = Some(now + Duration::from_secs(self.config.recovery_secs));
            tracing::warn!(
                tool = tool_name,
                failures = state.failures.len(),
                cooldown_secs = self.config.recovery_secs,
                "Circuit OPEN for tool"
            );
        }
    }

    /// Record a tool success. Clears failure history.
    pub fn record_success(&self, tool_name: &str) {
        let mut tools = self.tools.lock();
        tools.remove(tool_name);
    }

    /// Check if a tool's circuit is closed (available).
    pub fn is_available(&self, tool_name: &str) -> bool {
        let mut tools = self.tools.lock();
        if let Some(state) = tools.get_mut(tool_name) {
            if let Some(open_until) = state.circuit_open_until {
                if Instant::now() >= open_until {
                    // Circuit has recovered
                    state.circuit_open_until = None;
                    return true;
                }
                return false;
            }
        }
        true
    }

    /// Get health report for all tracked tools.
    pub fn health_report(&self) -> HashMap<String, String> {
        let now = Instant::now();
        let window = Duration::from_secs(self.config.failure_window_secs);
        let tools = self.tools.lock();
        let mut report = HashMap::new();

        for (name, state) in tools.iter() {
            let recent_failures = state
                .failures
                .iter()
                .filter(|t| now.duration_since(**t) < window)
                .count();

            if let Some(open_until) = state.circuit_open_until {
                if now < open_until {
                    let remaining = (open_until - now).as_secs();
                    report.insert(
                        name.clone(),
                        format!(
                            "UNAVAILABLE ({}s cooldown, {} failures)",
                            remaining, recent_failures
                        ),
                    );
                    continue;
                }
            }

            if recent_failures > 0 {
                report.insert(
                    name.clone(),
                    format!("degraded ({} recent failures)", recent_failures),
                );
            }
        }

        report
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_by_default() {
        let cb = CircuitBreaker::default();
        assert!(cb.is_available("web_search"));
    }

    #[test]
    fn test_circuit_opens_on_threshold() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            failure_window_secs: 60,
            recovery_secs: 60,
        });

        cb.record_failure("web_search");
        assert!(cb.is_available("web_search"));

        cb.record_failure("web_search");
        assert!(!cb.is_available("web_search"));
    }

    #[test]
    fn test_success_clears_failures() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        });

        cb.record_failure("web_search");
        cb.record_success("web_search");
        cb.record_failure("web_search");
        // Only 1 failure after reset, should still be available
        assert!(cb.is_available("web_search"));
    }

    #[test]
    fn test_health_report() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });

        cb.record_failure("web_search");
        let report = cb.health_report();
        assert!(report.get("web_search").unwrap().contains("degraded"));
    }
}
