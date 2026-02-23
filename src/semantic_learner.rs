//! Self-learning synonym discovery.
//!
//! Observes co-activated records and discovers synonym relationships
//! without external ML. Rewritten from aura-cognitive semantic_learner.py.

use std::collections::HashMap;
use crate::synonym::SynonymRing;

/// Configuration for the semantic learner.
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    /// Minimum co-occurrence count to consider a pair.
    pub min_cooccurrence: usize,
    /// Minimum word overlap ratio for candidate pairs.
    pub min_overlap: f32,
    /// Maximum number of pairs to track.
    pub max_pairs: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            min_cooccurrence: 3,
            min_overlap: 0.3,
            max_pairs: 10_000,
        }
    }
}

/// Self-supervised semantic learner.
pub struct SemanticLearnerEngine {
    /// Co-occurrence counts: (word_a, word_b) → count.
    cooccurrences: HashMap<(String, String), usize>,
    /// Configuration.
    config: LearnerConfig,
    /// Total learning cycles completed.
    pub cycle_count: u64,
}

impl SemanticLearnerEngine {
    pub fn new(config: LearnerConfig) -> Self {
        Self {
            cooccurrences: HashMap::new(),
            config,
            cycle_count: 0,
        }
    }

    /// Observe words from co-activated records.
    pub fn observe(&mut self, words_a: &[String], words_b: &[String]) {
        for wa in words_a {
            for wb in words_b {
                let a = wa.to_lowercase();
                let b = wb.to_lowercase();
                if a != b && a.len() > 2 && b.len() > 2 {
                    let key = if a < b { (a, b) } else { (b, a) };
                    *self.cooccurrences.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Evict least-seen pairs if over limit
        if self.cooccurrences.len() > self.config.max_pairs {
            let threshold = self.config.min_cooccurrence;
            self.cooccurrences.retain(|_, count| *count >= threshold);
        }
    }

    /// Extract confirmed synonym pairs and inject into the ring.
    pub fn extract_and_inject(&mut self, ring: &mut SynonymRing) -> usize {
        self.cycle_count += 1;
        let mut injected = 0;

        for ((a, b), count) in &self.cooccurrences {
            if *count >= self.config.min_cooccurrence && !ring.contains(a) {
                ring.add_pair(a, b);
                injected += 1;
            }
        }

        injected
    }

    /// Number of tracked pairs.
    pub fn tracked_pairs(&self) -> usize {
        self.cooccurrences.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_and_extract() {
        let mut learner = SemanticLearnerEngine::new(LearnerConfig {
            min_cooccurrence: 2,
            ..Default::default()
        });
        let mut ring = SynonymRing::new();

        let words_a = vec!["fast".into(), "performance".into()];
        let words_b = vec!["quick".into(), "speed".into()];

        // Observe twice to meet threshold
        learner.observe(&words_a, &words_b);
        learner.observe(&words_a, &words_b);

        let injected = learner.extract_and_inject(&mut ring);
        assert!(injected > 0);
    }
}
