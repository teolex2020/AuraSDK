//! Semantic SDR Enhancement Module
//!
//! Pure mathematical enhancements for SDR-based memory retrieval.
//! No external AI/LLM dependencies - uses only bitwise operations and Tanimoto similarity.
//!
//! # Components
//! - `synonym`: SDR union for synonym expansion
//! - `concepts`: Hierarchical bit inheritance (IS-A relations)
//! - `temporal`: Time-relative SDR encoding
//!
//! # Patent Claims Supported
//! - Claim 5: Synaptic Synthesis (T ≥ 0.75)
//! - Claim 7: O(k) SDR search complexity
//! - Claim 8: Entropy-weighted stability

pub mod synonym;
pub mod concepts;
pub mod temporal;

pub use synonym::SynonymExpander;
pub use concepts::ConceptGraph;
pub use temporal::TemporalResolver;

/// Semantic enhancement configuration
#[derive(Clone, Debug)]
pub struct SemanticConfig {
    /// Enable synonym expansion during retrieval
    pub expand_synonyms: bool,
    /// Enable concept hierarchy traversal
    pub use_concepts: bool,
    /// Enable temporal resolution
    pub resolve_temporal: bool,
    /// Maximum expansion depth for concepts
    pub max_concept_depth: usize,
    /// Synonym expansion limit per word
    pub max_synonyms_per_word: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            expand_synonyms: true,
            use_concepts: true,
            resolve_temporal: true,
            max_concept_depth: 3,
            max_synonyms_per_word: 5,
        }
    }
}

/// Unified semantic enhancer
///
/// Expands query text using synonyms, concept hierarchy, and temporal resolution.
/// Returns expanded text string which should be converted to SDR by the caller.
pub struct SemanticEnhancer {
    config: SemanticConfig,
    synonyms: SynonymExpander,
    concepts: ConceptGraph,
    temporal: TemporalResolver,
}

impl SemanticEnhancer {
    /// Create a new semantic enhancer with default configuration
    pub fn new() -> Self {
        Self::with_config(SemanticConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SemanticConfig) -> Self {
        Self {
            synonyms: SynonymExpander::new(),
            concepts: ConceptGraph::new(),
            temporal: TemporalResolver::new(),
            config,
        }
    }

    /// Expand query text with semantic enhancements
    ///
    /// # Math
    /// ```text
    /// text' = text ∪ Synonyms(text) ∪ Concepts(text)
    /// ```
    ///
    /// Returns expanded text. Caller should generate SDR from this text.
    pub fn expand_query(&self, text: &str) -> String {
        let mut expanded = String::new();

        // 1. Resolve temporal references first
        let resolved_text = if self.config.resolve_temporal {
            self.temporal.resolve(text)
        } else {
            text.to_string()
        };

        // Start with resolved text
        expanded.push_str(&resolved_text);
        expanded.push(' ');

        // 2. Expand with synonyms
        if self.config.expand_synonyms {
            let synonym_expansion = self.synonyms.expand_text(&resolved_text, self.config.max_synonyms_per_word);
            expanded.push_str(&synonym_expansion);
            expanded.push(' ');
        }

        // 3. Expand with concept hierarchy (upward traversal)
        if self.config.use_concepts {
            let concept_expansion = self.concepts.expand_upward(&resolved_text, self.config.max_concept_depth);
            expanded.push_str(&concept_expansion);
        }

        expanded.trim().to_string()
    }

    /// Get synonym expander for direct access
    pub fn synonyms(&self) -> &SynonymExpander {
        &self.synonyms
    }

    /// Get concept graph for direct access
    pub fn concepts(&self) -> &ConceptGraph {
        &self.concepts
    }

    /// Get temporal resolver for direct access
    pub fn temporal(&self) -> &TemporalResolver {
        &self.temporal
    }

    /// Add custom synonym pair
    pub fn add_synonym(&mut self, word: &str, synonym: &str) {
        self.synonyms.add_pair(word, synonym);
    }

    /// Add concept relationship (child IS-A parent)
    pub fn add_concept(&mut self, child: &str, parent: &str) {
        self.concepts.add_relation(child, parent);
    }

    /// Get configuration
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }
}

impl Default for SemanticEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_config_default() {
        let config = SemanticConfig::default();
        assert!(config.expand_synonyms);
        assert!(config.use_concepts);
        assert!(config.resolve_temporal);
    }

    #[test]
    fn test_semantic_enhancer_creation() {
        let enhancer = SemanticEnhancer::new();
        assert!(enhancer.config.expand_synonyms);
    }

    #[test]
    fn test_expand_query_with_synonyms() {
        let enhancer = SemanticEnhancer::new();
        let expanded = enhancer.expand_query("I bought a car");

        // Should contain synonyms for "car"
        assert!(expanded.contains("car"));
        assert!(expanded.contains("auto") || expanded.contains("vehicle"));
    }

    #[test]
    fn test_expand_query_with_concepts() {
        let enhancer = SemanticEnhancer::new();
        let expanded = enhancer.expand_query("I saw a dog");

        // Should contain ancestors for "dog"
        assert!(expanded.contains("dog"));
        assert!(expanded.contains("animal") || expanded.contains("mammal"));
    }

    #[test]
    fn test_expand_query_temporal() {
        let enhancer = SemanticEnhancer::new();
        let expanded = enhancer.expand_query("What happened yesterday");

        // Should resolve "yesterday" to a date
        assert!(!expanded.contains("yesterday") || expanded.contains("-")); // Contains date format
    }
}
