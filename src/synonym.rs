//! SynonymRing — semantic query expansion via synonym lookup.
//!
//! Rewritten from aura-cognitive synonym.py.
//! Complements CanonicalProjector: canonical normalizes input, SynonymRing expands queries.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Bidirectional synonym ring for query expansion.
#[derive(Debug, Clone)]
pub struct SynonymRing {
    /// word → set of synonyms (bidirectional).
    ring: HashMap<String, HashSet<String>>,
}

impl SynonymRing {
    pub fn new() -> Self {
        Self {
            ring: HashMap::new(),
        }
    }

    /// Add a pair of synonyms (bidirectional).
    pub fn add_pair(&mut self, a: &str, b: &str) {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        self.ring
            .entry(a_lower.clone())
            .or_default()
            .insert(b_lower.clone());
        self.ring.entry(b_lower).or_default().insert(a_lower);
    }

    /// Add a group of synonyms (all linked to each other).
    pub fn add_group(&mut self, words: &[&str]) {
        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                self.add_pair(words[i], words[j]);
            }
        }
    }

    /// Get synonyms for a word.
    pub fn get(&self, word: &str) -> Option<&HashSet<String>> {
        self.ring.get(&word.to_lowercase())
    }

    /// Expand a text by appending synonyms of each word.
    pub fn expand(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut expanded = words.iter().map(|w| w.to_string()).collect::<Vec<_>>();

        for word in &words {
            let lower = word.to_lowercase();
            if let Some(synonyms) = self.ring.get(&lower) {
                for syn in synonyms {
                    if !expanded.iter().any(|w| w.to_lowercase() == *syn) {
                        expanded.push(syn.clone());
                    }
                }
            }
        }

        expanded.join(" ")
    }

    /// Load synonyms from a TOML file.
    ///
    /// Expected format:
    /// ```toml
    /// [[groups]]
    /// words = ["fast", "quick", "rapid", "swift"]
    ///
    /// [[groups]]
    /// words = ["big", "large", "huge", "enormous"]
    /// ```
    pub fn load_toml(&mut self, path: &Path) -> Result<usize> {
        let content = std::fs::read_to_string(path)?;
        let value: toml::Value = content.parse()?;

        let mut count = 0;
        if let Some(groups) = value.get("groups").and_then(|v| v.as_array()) {
            for group in groups {
                if let Some(words) = group.get("words").and_then(|v| v.as_array()) {
                    let word_strs: Vec<&str> = words.iter().filter_map(|w| w.as_str()).collect();
                    self.add_group(&word_strs);
                    count += word_strs.len();
                }
            }
        }

        Ok(count)
    }

    /// Number of unique words in the ring.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check if a word has synonyms.
    pub fn contains(&self, word: &str) -> bool {
        self.ring.contains_key(&word.to_lowercase())
    }
}

impl Default for SynonymRing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_pair() {
        let mut ring = SynonymRing::new();
        ring.add_pair("fast", "quick");

        let syns = ring.get("fast").unwrap();
        assert!(syns.contains("quick"));

        let syns = ring.get("quick").unwrap();
        assert!(syns.contains("fast"));
    }

    #[test]
    fn test_add_group() {
        let mut ring = SynonymRing::new();
        ring.add_group(&["big", "large", "huge"]);

        let syns = ring.get("big").unwrap();
        assert!(syns.contains("large"));
        assert!(syns.contains("huge"));
    }

    #[test]
    fn test_expand() {
        let mut ring = SynonymRing::new();
        ring.add_pair("fast", "quick");

        let expanded = ring.expand("fast car");
        assert!(expanded.contains("fast"));
        assert!(expanded.contains("quick"));
        assert!(expanded.contains("car"));
    }

    #[test]
    fn test_case_insensitive() {
        let mut ring = SynonymRing::new();
        ring.add_pair("Fast", "Quick");
        assert!(ring.contains("fast"));
        assert!(ring.contains("QUICK"));
    }
}
