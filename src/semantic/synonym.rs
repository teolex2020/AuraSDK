//! Synonym SDR Expansion
//!
//! Expands query SDRs by including synonyms using pure set operations.
//!
//! # Math
//! ```text
//! SDR_expanded = SDR_query ∪ {SDR(s) | s ∈ synonyms(word) for word in query}
//! ```
//!
//! # Example
//! Query: "car" → Expanded: "car" ∪ "auto" ∪ "vehicle" ∪ "automobile"

use std::collections::HashMap;

/// Synonym expander using precomputed word pairs
pub struct SynonymExpander {
    /// word -> list of synonyms
    synonyms: HashMap<String, Vec<String>>,
    /// Bidirectional flag (if true, A↔B means both directions)
    bidirectional: bool,
}

impl SynonymExpander {
    /// Create a new synonym expander with default lexicon
    pub fn new() -> Self {
        let mut expander = Self {
            synonyms: HashMap::new(),
            bidirectional: true,
        };
        expander.load_default_lexicon();
        expander
    }

    /// Create empty expander (for custom synonyms only)
    pub fn empty() -> Self {
        Self {
            synonyms: HashMap::new(),
            bidirectional: true,
        }
    }

    /// Load default English synonym lexicon (~500 common pairs)
    fn load_default_lexicon(&mut self) {
        // Common synonym groups - each word maps to others in group
        let groups: &[&[&str]] = &[
            // Vehicles
            &["car", "auto", "automobile", "vehicle"],
            &["truck", "lorry", "pickup"],
            &["bike", "bicycle", "cycle"],
            &["plane", "airplane", "aircraft", "jet"],
            &["boat", "ship", "vessel"],

            // People
            &["person", "human", "individual", "people"],
            &["child", "kid", "youngster"],
            &["man", "male", "gentleman"],
            &["woman", "female", "lady"],
            &["doctor", "physician", "medic"],
            &["teacher", "instructor", "educator"],

            // Actions
            &["run", "sprint", "dash", "jog"],
            &["walk", "stroll", "amble"],
            &["talk", "speak", "chat", "converse"],
            &["eat", "consume", "dine"],
            &["drink", "sip", "gulp"],
            &["sleep", "rest", "slumber", "nap"],
            &["work", "labor", "toil"],
            &["play", "frolic", "recreation"],
            &["think", "ponder", "contemplate", "consider"],
            &["create", "make", "build", "construct"],
            &["destroy", "demolish", "ruin"],
            &["start", "begin", "commence", "initiate"],
            &["stop", "end", "finish", "conclude", "terminate"],
            &["help", "assist", "aid", "support"],
            &["buy", "purchase", "acquire"],
            &["sell", "vend", "trade"],

            // Emotions
            &["happy", "joyful", "glad", "pleased", "delighted"],
            &["sad", "unhappy", "sorrowful", "melancholy"],
            &["angry", "mad", "furious", "irate"],
            &["scared", "afraid", "frightened", "terrified"],
            &["love", "adore", "cherish"],
            &["hate", "despise", "loathe", "detest"],

            // Sizes
            &["big", "large", "huge", "enormous", "massive"],
            &["small", "little", "tiny", "miniature"],
            &["tall", "high", "lofty"],
            &["short", "low", "brief"],
            &["wide", "broad", "expansive"],
            &["narrow", "thin", "slim"],

            // Qualities
            &["good", "great", "excellent", "fine", "superb"],
            &["bad", "poor", "terrible", "awful"],
            &["fast", "quick", "rapid", "swift", "speedy"],
            &["slow", "sluggish", "leisurely"],
            &["hot", "warm", "heated"],
            &["cold", "cool", "chilly", "freezing"],
            &["new", "fresh", "novel", "recent"],
            &["old", "ancient", "aged", "vintage"],
            &["easy", "simple", "effortless"],
            &["hard", "difficult", "challenging", "tough"],
            &["smart", "intelligent", "clever", "brilliant"],
            &["stupid", "dumb", "foolish"],
            &["beautiful", "pretty", "gorgeous", "lovely", "attractive"],
            &["ugly", "unattractive", "hideous"],
            &["rich", "wealthy", "affluent"],
            &["poor", "impoverished", "needy"],
            &["strong", "powerful", "mighty", "robust"],
            &["weak", "feeble", "frail"],
            &["clean", "pure", "spotless"],
            &["dirty", "filthy", "grimy"],

            // Places
            &["house", "home", "residence", "dwelling"],
            &["store", "shop", "market"],
            &["office", "workplace", "bureau"],
            &["school", "academy", "institution"],
            &["hospital", "clinic", "infirmary"],
            &["road", "street", "path", "way"],
            &["city", "town", "metropolis"],
            &["country", "nation", "land"],

            // Time
            &["now", "currently", "presently"],
            &["soon", "shortly", "promptly"],
            &["later", "afterward", "subsequently"],
            &["always", "forever", "eternally"],
            &["never", "not ever"],

            // Tech/Computing
            &["computer", "pc", "machine"],
            &["program", "software", "application", "app"],
            &["data", "information", "content"],
            &["file", "document", "record"],
            &["error", "bug", "fault", "defect"],
            &["fix", "repair", "patch", "resolve"],
            &["code", "program", "script"],
            &["memory", "ram", "storage"],
            &["database", "db", "datastore"],
            &["server", "host", "backend"],
            &["client", "frontend", "user"],
            &["network", "net", "connection"],
            &["internet", "web", "online"],

            // Business
            &["money", "cash", "funds", "currency"],
            &["job", "work", "employment", "occupation"],
            &["company", "business", "firm", "corporation"],
            &["customer", "client", "buyer"],
            &["product", "item", "goods"],
            &["price", "cost", "value"],
            &["profit", "gain", "earnings"],
            &["loss", "deficit", "shortfall"],
        ];

        for group in groups {
            self.add_group(group);
        }
    }

    /// Add a group of synonyms (all words map to each other)
    pub fn add_group(&mut self, words: &[&str]) {
        for word in words {
            let key = word.to_lowercase();
            let synonyms: Vec<String> = words
                .iter()
                .filter(|w| **w != *word)
                .map(|w| w.to_lowercase())
                .collect();

            self.synonyms
                .entry(key)
                .or_default()
                .extend(synonyms);
        }
    }

    /// Add a single synonym pair
    pub fn add_pair(&mut self, word_a: &str, word_b: &str) {
        let a = word_a.to_lowercase();
        let b = word_b.to_lowercase();

        self.synonyms
            .entry(a.clone())
            .or_default()
            .push(b.clone());

        if self.bidirectional {
            self.synonyms
                .entry(b)
                .or_default()
                .push(a);
        }
    }

    /// Get synonyms for a word
    pub fn get_synonyms(&self, word: &str) -> Vec<&str> {
        self.synonyms
            .get(&word.to_lowercase())
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Expand text with synonyms, returning expanded text string
    ///
    /// # Math
    /// For each word w in text: text' = text ∪ {synonym | synonym ∈ synonyms(w)}
    ///
    /// The caller should generate SDR from the expanded text.
    pub fn expand_text(&self, text: &str, max_per_word: usize) -> String {
        let mut expanded_text = String::new();

        for word in text.split_whitespace() {
            // Add original word
            expanded_text.push_str(word);
            expanded_text.push(' ');

            // Add synonyms (up to limit)
            let synonyms = self.get_synonyms(word);
            for syn in synonyms.iter().take(max_per_word) {
                expanded_text.push_str(syn);
                expanded_text.push(' ');
            }
        }

        expanded_text.trim().to_string()
    }

    /// Check if two words are synonyms
    pub fn are_synonyms(&self, word_a: &str, word_b: &str) -> bool {
        let a = word_a.to_lowercase();
        let b = word_b.to_lowercase();

        if a == b {
            return true;
        }

        self.synonyms
            .get(&a)
            .map(|syns| syns.contains(&b))
            .unwrap_or(false)
    }

    /// Get total number of synonym entries
    pub fn len(&self) -> usize {
        self.synonyms.len()
    }

    /// Check if lexicon is empty
    pub fn is_empty(&self) -> bool {
        self.synonyms.is_empty()
    }

    /// Get all words in lexicon
    pub fn words(&self) -> Vec<&str> {
        self.synonyms.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SynonymExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_lookup() {
        let expander = SynonymExpander::new();

        let syns = expander.get_synonyms("car");
        assert!(syns.contains(&"auto"));
        assert!(syns.contains(&"vehicle"));
    }

    #[test]
    fn test_are_synonyms() {
        let expander = SynonymExpander::new();

        assert!(expander.are_synonyms("car", "auto"));
        assert!(expander.are_synonyms("happy", "joyful"));
        assert!(!expander.are_synonyms("car", "happy"));
    }

    #[test]
    fn test_bidirectional() {
        let expander = SynonymExpander::new();

        // If car -> auto, then auto -> car
        let car_syns = expander.get_synonyms("car");
        let auto_syns = expander.get_synonyms("auto");

        assert!(car_syns.contains(&"auto"));
        assert!(auto_syns.contains(&"car"));
    }

    #[test]
    fn test_custom_synonym() {
        let mut expander = SynonymExpander::empty();
        expander.add_pair("aura", "memory");

        assert!(expander.are_synonyms("aura", "memory"));
        assert!(expander.are_synonyms("memory", "aura")); // bidirectional
    }

    #[test]
    fn test_lexicon_size() {
        let expander = SynonymExpander::new();
        assert!(expander.len() > 100, "Should have at least 100 words");
    }
}
