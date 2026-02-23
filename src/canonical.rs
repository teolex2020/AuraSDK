//! Canonical Projection — v2.0 Semantic Enhancement
//!
//! Normalizes text through synonym maps before SDR generation.
//! "kitten" → "cat", "кіт" → "cat", "café" → "cafe"
//!
//! # Architecture
//! - Static, deterministic HashMap<String, String> — zero neural computation
//! - O(1) lookup per token, < 5µs overhead per text
//! - Loaded from binary `.aura.syn` files (compiled from TOML source)
//! - Optional: if no synonym file exists, pipeline is unchanged (v1.8 behavior)

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use anyhow::Result;

/// Canonical projector — maps words to their canonical forms.
///
/// Example usage:
/// ```ignore
/// let projector = CanonicalProjector::from_groups(&[
///     vec!["cat", "kitten", "kitty", "feline"],
///     vec!["car", "auto", "automobile"],
/// ]);
/// assert_eq!(projector.project("the kitten drives an automobile"), "the cat drives an car");
/// ```
pub struct CanonicalProjector {
    /// word → canonical form (e.g., "kitten" → "cat")
    map: HashMap<String, String>,
    /// Hit counter (tokens that matched)
    hit_count: AtomicU64,
    /// Miss counter (tokens that didn't match)
    miss_count: AtomicU64,
}

impl CanonicalProjector {
    /// Create an empty projector (no synonyms loaded).
    pub fn empty() -> Self {
        Self {
            map: HashMap::new(),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Build from synonym groups. First word in each group is the canonical form.
    ///
    /// Example: `["cat", "kitten", "kitty"]` → kitten→cat, kitty→cat
    pub fn from_groups(groups: &[Vec<&str>]) -> Self {
        let mut map = HashMap::new();
        for group in groups {
            if group.len() < 2 {
                continue;
            }
            let canonical = group[0].to_lowercase();
            for word in &group[1..] {
                let key = word.to_lowercase();
                if key != canonical {
                    map.insert(key, canonical.clone());
                }
            }
        }
        Self {
            map,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Build from explicit key→value pairs.
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut map = HashMap::new();
        for (from, to) in pairs {
            map.insert(from.to_lowercase(), to.to_lowercase());
        }
        Self {
            map,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Merge another projector's map into this one.
    /// Existing keys are NOT overwritten (first definition wins).
    pub fn merge(&mut self, other: &CanonicalProjector) {
        for (k, v) in &other.map {
            self.map.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    /// Insert a single mapping.
    pub fn insert(&mut self, from: &str, to: &str) {
        self.map.insert(from.to_lowercase(), to.to_lowercase());
    }

    /// Load from a bincode-serialized binary file (.aura.syn).
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let map: HashMap<String, String> = bincode::deserialize(&data)?;
        Ok(Self {
            map,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        })
    }

    /// Save to a bincode-serialized binary file (.aura.syn).
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = bincode::serialize(&self.map)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Number of entries in the map.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get hit/miss statistics.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hit_count.load(Ordering::Relaxed),
            self.miss_count.load(Ordering::Relaxed),
        )
    }

    /// Project a single token to its canonical form.
    /// Returns the canonical form if found, otherwise the original (lowercased).
    #[inline]
    pub fn project_token<'a>(&'a self, token: &str) -> Option<&'a str> {
        // Lowercase + strip diacritics for lookup
        let key = strip_diacritics(&token.to_lowercase());
        self.map.get(key.as_str()).map(|s| s.as_str())
    }

    /// Project entire text: split on whitespace, project each token, rejoin.
    ///
    /// Original text structure (spacing) is normalized to single spaces.
    /// Only the SDR sees the projected text — original text is always stored as-is.
    pub fn project(&self, text: &str) -> String {
        if text.is_empty() || self.map.is_empty() {
            return text.to_string();
        }

        let mut result = String::with_capacity(text.len());
        let mut first = true;

        for token in text.split_whitespace() {
            if !first {
                result.push(' ');
            }
            first = false;

            let lower = token.to_lowercase();
            let stripped = strip_diacritics(&lower);

            if let Some(canonical) = self.map.get(stripped.as_str()) {
                result.push_str(canonical);
                self.hit_count.fetch_add(1, Ordering::Relaxed);
            } else {
                // Keep original (lowercased) token
                result.push_str(&lower);
                self.miss_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        result
    }
}

/// Strip common diacritical marks from a string.
///
/// Uses char-by-char replacement for common accented Latin characters.
/// Covers: à-ÿ range (French, German, Spanish, Portuguese, etc.)
/// Does NOT touch Cyrillic, CJK, or other scripts — those pass through as-is.
pub fn strip_diacritics(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        result.push(fold_diacritic(ch));
    }
    result
}

/// Fold a single accented character to its ASCII base.
#[inline]
fn fold_diacritic(ch: char) -> char {
    match ch {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'æ' => 'a', // simplification
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ð' => 'd',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ø' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        'þ' => 't', // simplification
        // Uppercase variants
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'a',
        'Æ' => 'a',
        'Ç' => 'c',
        'È' | 'É' | 'Ê' | 'Ë' => 'e',
        'Ì' | 'Í' | 'Î' | 'Ï' => 'i',
        'Ð' => 'd',
        'Ñ' => 'n',
        'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'o',
        'Ø' => 'o',
        'Ù' | 'Ú' | 'Û' | 'Ü' => 'u',
        'Ý' => 'y',
        'Þ' => 't',
        _ => ch,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_projector() -> CanonicalProjector {
        let mut p = CanonicalProjector::from_groups(&[
            vec!["cat", "kitten", "kitty", "feline"],
            vec!["car", "auto", "automobile", "vehicle"],
            vec!["run", "sprint", "dash", "jog"],
            vec!["eat", "consume", "dine"],
        ]);
        // Add cross-lingual
        p.insert("кіт", "cat");
        p.insert("кішка", "cat");
        p.insert("машина", "car");
        p
    }

    #[test]
    fn test_project_synonym_basic() {
        let p = test_projector();
        // "kitten" → "cat", "eats" stays (not in synonym map — only "eat"/"consume"/"dine")
        assert_eq!(p.project("the kitten eats fish"), "the cat eats fish");
        // "consume" → "eat"
        assert_eq!(p.project("the kitten consume fish"), "the cat eat fish");
    }

    #[test]
    fn test_project_unknown_passthrough() {
        let p = test_projector();
        assert_eq!(p.project("unknown words here"), "unknown words here");
    }

    #[test]
    fn test_project_empty() {
        let p = test_projector();
        assert_eq!(p.project(""), "");
    }

    #[test]
    fn test_project_preserves_structure() {
        let p = test_projector();
        // Multiple words, some projected, some not
        let result = p.project("a kitten and an automobile");
        assert_eq!(result, "a cat and an car");
    }

    #[test]
    fn test_diacritic_removal() {
        assert_eq!(strip_diacritics("café"), "cafe");
        assert_eq!(strip_diacritics("naïve"), "naive");
        assert_eq!(strip_diacritics("über"), "uber");
        assert_eq!(strip_diacritics("résumé"), "resume");
    }

    #[test]
    fn test_cross_lingual() {
        let p = test_projector();
        assert_eq!(p.project("кіт"), "cat");
        assert_eq!(p.project("кішка"), "cat");
        assert_eq!(p.project("машина"), "car");
    }

    #[test]
    fn test_from_groups() {
        let p = CanonicalProjector::from_groups(&[
            vec!["alpha", "beta", "gamma"],
        ]);
        assert_eq!(p.len(), 2); // beta→alpha, gamma→alpha
        assert_eq!(p.project("beta"), "alpha");
        assert_eq!(p.project("gamma"), "alpha");
        assert_eq!(p.project("alpha"), "alpha"); // canonical passes through
    }

    #[test]
    fn test_save_load_roundtrip() {
        let p = test_projector();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.aura.syn");

        p.save(&path).unwrap();
        let loaded = CanonicalProjector::load(&path).unwrap();

        assert_eq!(loaded.len(), p.len());
        assert_eq!(loaded.project("kitten"), p.project("kitten"));
        assert_eq!(loaded.project("кіт"), p.project("кіт"));
    }

    #[test]
    fn test_case_insensitive() {
        let p = test_projector();
        assert_eq!(p.project("Kitten"), "cat");
        assert_eq!(p.project("KITTEN"), "cat");
        assert_eq!(p.project("KiTtEn"), "cat");
    }

    #[test]
    fn test_tanimoto_improvement() {
        use crate::sdr::SDRInterpreter;

        let sdr = SDRInterpreter::default();
        let p = test_projector();

        // Without projection: cat ≠ kitten
        let sdr_cat_raw = sdr.text_to_sdr("cat", false);
        let sdr_kitten_raw = sdr.text_to_sdr("kitten", false);
        let raw_tanimoto = sdr.tanimoto_sparse(&sdr_cat_raw, &sdr_kitten_raw);

        // With projection: both become "cat"
        let projected_cat = p.project("cat");
        let projected_kitten = p.project("kitten");
        let sdr_cat_proj = sdr.text_to_sdr(&projected_cat, false);
        let sdr_kitten_proj = sdr.text_to_sdr(&projected_kitten, false);
        let proj_tanimoto = sdr.tanimoto_sparse(&sdr_cat_proj, &sdr_kitten_proj);

        // Raw should be very low (structural mismatch)
        assert!(raw_tanimoto < 0.1, "Raw tanimoto should be near 0, got {}", raw_tanimoto);
        // Projected should be identical (both → "cat")
        assert!(proj_tanimoto > 0.99, "Projected tanimoto should be ~1.0, got {}", proj_tanimoto);
    }

    #[test]
    fn test_stats() {
        let p = test_projector();
        p.project("kitten unknown");
        let (hits, misses) = p.stats();
        assert_eq!(hits, 1); // kitten → cat
        assert_eq!(misses, 1); // unknown
    }

    #[test]
    fn test_merge() {
        let mut p1 = CanonicalProjector::from_pairs(&[("a", "x")]);
        let p2 = CanonicalProjector::from_pairs(&[("b", "y"), ("a", "z")]);
        p1.merge(&p2);
        assert_eq!(p1.project("a"), "x"); // first definition wins
        assert_eq!(p1.project("b"), "y"); // merged from p2
    }
}
