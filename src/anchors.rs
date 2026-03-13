use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorData {
    pub text: String,
    pub category: String,
    pub dna: String,
    pub impact: f32,
    pub crystallization_trigger: String, // "explicit", "identity", "intensity", "structural"
}

/// Anchor detection via Cognitive Crystallization.
///
/// Anchors are created when:
/// 1. Explicitly marked via API (force_pin parameter)
/// 2. Intensity exceeds crystallization threshold (3.5)
/// 3. Structural significance (long, detailed content with metrics)
/// 4. SFT text markers ("Anchor:", "Important:", "Critical:", "Identity:")
pub struct AnchorManager {
    crystallization_threshold: f32,
    min_text_length: usize,
    metrics_re: Regex,
}

impl Default for AnchorManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AnchorManager {
    pub fn new() -> Self {
        Self {
            crystallization_threshold: 3.5,
            min_text_length: 20,
            metrics_re: Regex::new(r"(?i)\d+\.?\d*\s*(ms|%|gb|mb|kb|fps|hz)")
                .expect("invalid metrics regex"),
        }
    }

    pub fn evaluate_impact(&self, text: &str) -> f32 {
        let mut impact_score = 0.0;

        // 1. Structural weight (length)
        let char_count = text.chars().count();
        if char_count > 150 {
            impact_score += 1.0;
        } else if char_count > 80 {
            impact_score += 0.5;
        }

        // 2. Numeric content (dates, metrics)
        let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
        if digit_count >= 2 {
            impact_score += 0.3;
        }

        // 3. Technical precision: mentions of units/metrics (ms, %, GB, etc.)
        if self.metrics_re.is_match(text) {
            impact_score += 0.5;
        }

        impact_score
    }

    /// Determine if a memory should become an Anchor (user_core).
    ///
    /// Crystallization criteria:
    /// - force_pin=true: Explicit API request
    /// - intensity >= 3.5: High-salience content
    /// - Structural significance: Long, detailed content with metrics (impact >= 1.3)
    /// - SFT markers: Explicit text prefixes ("Anchor:", "Important:", etc.)
    pub fn evaluate_and_pin(
        &self,
        text: &str,
        intensity: f32,
        force_pin: bool,
    ) -> Option<AnchorData> {
        if text.len() < self.min_text_length {
            return None;
        }

        let impact = self.evaluate_impact(text);

        let is_explicit_pin = force_pin;
        let is_crystallized = intensity >= self.crystallization_threshold;
        let is_structurally_significant = impact >= 1.3;

        // SFT markers - explicit text prefixes that force anchor creation
        let sft_markers = ["Anchor:", "Important:", "Critical:", "Identity:"];
        let is_sft_marked = sft_markers.iter().any(|m| text.contains(m));

        if is_explicit_pin || is_crystallized || is_structurally_significant || is_sft_marked {
            let trigger = if is_explicit_pin {
                "explicit"
            } else if is_crystallized {
                "intensity"
            } else {
                "structural"
            };

            return Some(AnchorData {
                text: text.to_string(),
                category: "crystallized".to_string(),
                dna: "user_core".to_string(),
                impact,
                crystallization_trigger: trigger.to_string(),
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_impact() {
        let manager = AnchorManager::new();

        let impact1 = manager.evaluate_impact("Our latency is 0.29ms");
        let impact2 = manager.evaluate_impact("Hello world");

        assert!(impact1 > impact2, "Metrics should boost impact");
    }
}
