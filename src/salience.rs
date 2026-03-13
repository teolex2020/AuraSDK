use regex::Regex;
use std::collections::HashMap;

/// Enhanced Salience Scorer - Emotional Resonance Module.
///
/// Determines the "weight" of a memory based on multi-factor analysis:
/// - Emotional intensity (I)
/// - Information entropy/stability (S)
/// - Cross-context resonance (R)
///
/// # Math (Patent Claim 8)
/// ```text
/// Salience Ψ = α×I + β×S + γ×R
/// ```
/// where α, β, γ are tunable weights (default: 0.5, 0.3, 0.2)
#[allow(dead_code)]
pub struct SalienceScorer {
    identity_patterns: Vec<Regex>,
    factual_patterns: Vec<Regex>,
    intensifier_patterns: Vec<Regex>,
    emotional_patterns: Vec<(Regex, f32)>, // (pattern, weight)

    // Weights for the final formula
    alpha: f32, // Intensity weight
    beta: f32,  // Entropy weight
    gamma: f32, // Resonance weight
}

impl Default for SalienceScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl SalienceScorer {
    pub fn new() -> Self {
        Self {
            identity_patterns: vec![
                Regex::new(r"(?i)\b(я|мене|мені|моє|мій|моя|мої)\b").unwrap(),
                Regex::new(r"(?i)\b(i|me|my|mine|am)\b").unwrap(),
            ],
            factual_patterns: vec![
                Regex::new(r"\d+").unwrap(),
                Regex::new(r"(?i)\b(рік|року|років|year|years)\b").unwrap(),
                Regex::new(r"(?i)\b(народив|born)\b").unwrap(),
            ],
            intensifier_patterns: vec![
                Regex::new(r"(?i)\b(завжди|ніколи|обожнюю|ненавиджу|важливо|критично)\b").unwrap(),
                Regex::new(r"(?i)\b(always|never|love|hate|important|critical)\b").unwrap(),
                Regex::new(r"(?i)\b(дуже|very)\b").unwrap(),
            ],
            emotional_patterns: vec![
                // High positive emotions
                (
                    Regex::new(r"(?i)\b(love|adore|cherish|treasure)\b").unwrap(),
                    2.0,
                ),
                (
                    Regex::new(r"(?i)\b(happy|joy|excited|thrilled)\b").unwrap(),
                    1.5,
                ),
                (
                    Regex::new(r"(?i)\b(кохаю|люблю|щасливий|радий)\b").unwrap(),
                    2.0,
                ),
                // High negative emotions (also high salience)
                (
                    Regex::new(r"(?i)\b(hate|despise|fear|terrified)\b").unwrap(),
                    2.0,
                ),
                (
                    Regex::new(r"(?i)\b(angry|furious|devastated)\b").unwrap(),
                    1.8,
                ),
                (
                    Regex::new(r"(?i)\b(ненавиджу|боюся|злий|сумний)\b").unwrap(),
                    2.0,
                ),
                // Safety-critical
                (
                    Regex::new(r"(?i)\b(danger|emergency|urgent|critical)\b").unwrap(),
                    3.0,
                ),
                (
                    Regex::new(r"(?i)\b(небезпека|терміново|критично)\b").unwrap(),
                    3.0,
                ),
                // Strategic importance
                (
                    Regex::new(r"(?i)\b(secret|confidential|password|key)\b").unwrap(),
                    2.5,
                ),
                (Regex::new(r"(?i)\b(таємниця|пароль|ключ)\b").unwrap(), 2.5),
            ],
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
        }
    }

    /// Create scorer with custom weights
    #[allow(dead_code)]
    pub fn with_weights(alpha: f32, beta: f32, gamma: f32) -> Self {
        let mut scorer = Self::new();
        scorer.alpha = alpha;
        scorer.beta = beta;
        scorer.gamma = gamma;
        scorer
    }

    /// Legacy scoring method (for compatibility)
    pub fn score_text(&self, text: &str) -> f32 {
        let mut score = 0.5; // Base score

        // 1. Identity Check (+1.0)
        for pattern in &self.identity_patterns {
            if pattern.is_match(text) {
                score += 1.0;
                break;
            }
        }

        // 2. Factual Check (+1.5)
        // Check for numbers
        if self.factual_patterns[0].is_match(text) {
            score += 1.5;
        }

        // 3. Intensifier Check (+0.5 per match, max 1.0)
        let mut intensifier_count = 0;
        for pattern in &self.intensifier_patterns {
            if pattern.is_match(text) {
                intensifier_count += 1;
            }
        }
        score += (intensifier_count as f32 * 0.5).min(1.0);

        (score * 100.0).round() / 100.0
    }

    /// Enhanced scoring with full formula
    ///
    /// # Math
    /// Ψ = α×I + β×S + γ×R
    ///
    /// Returns value in [0.0, 10.0] range
    #[allow(dead_code)]
    pub fn compute_salience(&self, text: &str) -> SalienceResult {
        let intensity = self.compute_intensity(text);
        let entropy = self.compute_entropy(text);
        let resonance = 1.0; // Base resonance (requires cross-memory context)

        let psi = self.alpha * intensity + self.beta * entropy + self.gamma * resonance;

        // Scale to [0, 10] range
        let normalized = (psi * 2.5).clamp(0.0, 10.0);

        SalienceResult {
            total: (normalized * 100.0).round() / 100.0,
            intensity,
            entropy,
            resonance,
        }
    }

    /// Compute emotional intensity score (I)
    ///
    /// # Math
    /// I = Σ(w_i × count_i) for each emotional pattern
    #[allow(dead_code)]
    pub fn compute_intensity(&self, text: &str) -> f32 {
        let mut intensity = 0.0;

        // Check identity patterns (+1.0)
        for pattern in &self.identity_patterns {
            if pattern.is_match(text) {
                intensity += 1.0;
                break;
            }
        }

        // Check factual patterns (+1.0)
        for pattern in &self.factual_patterns {
            if pattern.is_match(text) {
                intensity += 1.0;
                break;
            }
        }

        // Check emotional patterns with weights
        for (pattern, weight) in &self.emotional_patterns {
            let count = pattern.find_iter(text).count() as f32;
            intensity += count * weight;
        }

        // Check intensifiers (+0.5 each, max 2.0)
        let mut intensifier_score = 0.0;
        for pattern in &self.intensifier_patterns {
            let count = pattern.find_iter(text).count() as f32;
            intensifier_score += count * 0.5;
        }
        intensity += intensifier_score.min(2.0);

        intensity
    }

    /// Compute information entropy/stability (S)
    ///
    /// # Math
    /// S = -Σ(p_i × log2(p_i)) where p_i = freq(word_i) / total_words
    ///
    /// Higher entropy = more information-dense = higher stability
    #[allow(dead_code)]
    pub fn compute_entropy(&self, text: &str) -> f32 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Count word frequencies
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *freq.entry(*word).or_insert(0) += 1;
        }

        // Compute Shannon entropy
        let total = words.len() as f32;
        let mut entropy = 0.0;

        for count in freq.values() {
            let p = *count as f32 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        // Normalize by max possible entropy (log2(n) where n = unique words)
        let max_entropy = (freq.len() as f32).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Compute cross-context resonance (R) given related memories
    ///
    /// # Math
    /// R = Π(Tanimoto_i)^(1/n) (geometric mean of similarities)
    ///
    /// Higher resonance = appears in multiple contexts = more important
    #[allow(dead_code)]
    pub fn compute_resonance(&self, similarities: &[f32]) -> f32 {
        if similarities.is_empty() {
            return 1.0; // Default resonance
        }

        // Geometric mean of similarities
        let n = similarities.len() as f32;
        let product: f32 = similarities.iter().filter(|&&s| s > 0.0).product();

        if product > 0.0 {
            product.powf(1.0 / n)
        } else {
            0.0
        }
    }

    /// Check if text contains safety-critical content
    #[allow(dead_code)]
    pub fn is_safety_critical(&self, text: &str) -> bool {
        for (pattern, weight) in &self.emotional_patterns {
            if *weight >= 2.5 && pattern.is_match(text) {
                return true;
            }
        }
        false
    }
}

/// Result of salience computation
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SalienceResult {
    /// Total salience score [0.0, 10.0]
    pub total: f32,
    /// Emotional intensity component
    pub intensity: f32,
    /// Information entropy component
    pub entropy: f32,
    /// Cross-context resonance component
    pub resonance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salience_scoring() {
        let scorer = SalienceScorer::new();

        // "I was born in 1990" -> Identity(1.0) + Number(1.5) + Base(0.5) = 3.0
        let s1 = scorer.score_text("I was born in 1990");
        assert!(s1 >= 3.0);

        // "Hello world" -> Base(0.5)
        let s2 = scorer.score_text("Hello world");
        assert_eq!(s2, 0.5);
    }

    #[test]
    fn test_compute_intensity() {
        let scorer = SalienceScorer::new();

        // High emotional content
        let i1 = scorer.compute_intensity("I love this so much, it's critical!");
        let i2 = scorer.compute_intensity("Hello world");

        assert!(i1 > i2, "Emotional text should have higher intensity");
        assert!(i1 > 2.0, "Love + critical should be high intensity");
    }

    #[test]
    fn test_compute_entropy() {
        let scorer = SalienceScorer::new();

        // High entropy (many unique words)
        let e1 = scorer.compute_entropy("The quick brown fox jumps over the lazy dog");
        // Low entropy (repetitive)
        let e2 = scorer.compute_entropy("yes yes yes yes yes");

        assert!(e1 > e2, "Diverse text should have higher entropy");
        assert!(e1 > 0.5, "Sentence should have reasonable entropy");
    }

    #[test]
    fn test_compute_resonance() {
        let scorer = SalienceScorer::new();

        // Multiple strong matches
        let r1 = scorer.compute_resonance(&[0.8, 0.7, 0.9]);
        // Single weak match
        let r2 = scorer.compute_resonance(&[0.2]);

        assert!(r1 > r2, "Strong matches should have higher resonance");
    }

    #[test]
    fn test_compute_salience() {
        let scorer = SalienceScorer::new();

        // Safety-critical content
        let s1 = scorer.compute_salience("DANGER! This is an emergency situation!");
        // Neutral content
        let s2 = scorer.compute_salience("The weather is nice today");

        assert!(s1.total > s2.total, "Critical content should score higher");
        assert!(s1.total >= 2.0, "Emergency should have high salience");
    }

    #[test]
    fn test_is_safety_critical() {
        let scorer = SalienceScorer::new();

        assert!(scorer.is_safety_critical("DANGER ahead!"));
        assert!(scorer.is_safety_critical("This is my secret password"));
        assert!(!scorer.is_safety_critical("Hello world"));
    }

    #[test]
    fn test_custom_weights() {
        let scorer = SalienceScorer::with_weights(0.8, 0.1, 0.1);

        // With high alpha, intensity should dominate
        let result = scorer.compute_salience("I love this!");
        assert!(result.total > 0.0);
    }
}
