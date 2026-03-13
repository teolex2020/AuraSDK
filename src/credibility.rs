//! Source credibility — domain reputation scoring.
//!
//! Rewritten from source_credibility.py.

use std::collections::HashMap;

/// Default credibility score for unknown domains.
const DEFAULT_SCORE: f32 = 0.50;

/// Source credibility scorer with domain reputation table.
pub struct SourceCredibility {
    /// Domain → credibility score (0.0-1.0).
    cache: HashMap<String, f32>,
    /// User overrides take priority.
    user_overrides: HashMap<String, f32>,
}

impl SourceCredibility {
    pub fn new() -> Self {
        let mut cache = HashMap::new();

        // Medical / Scientific (High Trust)
        cache.insert("pubmed.ncbi.nlm.nih.gov".into(), 0.95);
        cache.insert("ncbi.nlm.nih.gov".into(), 0.95);
        cache.insert("who.int".into(), 0.95);
        cache.insert("cdc.gov".into(), 0.95);
        cache.insert("mayoclinic.org".into(), 0.90);
        cache.insert("clevelandclinic.org".into(), 0.90);
        cache.insert("hopkinsmedicine.org".into(), 0.90);
        cache.insert("medlineplus.gov".into(), 0.90);
        cache.insert("sciencedirect.com".into(), 0.85);
        cache.insert("nature.com".into(), 0.90);
        cache.insert("bmj.com".into(), 0.90);
        cache.insert("thelancet.com".into(), 0.90);
        cache.insert("webmd.com".into(), 0.70);
        cache.insert("healthline.com".into(), 0.65);
        cache.insert("medicalnewstoday.com".into(), 0.65);

        // Academic / Reference (High Trust)
        cache.insert("arxiv.org".into(), 0.90);
        cache.insert("scholar.google.com".into(), 0.85);
        cache.insert("wikipedia.org".into(), 0.75);
        cache.insert("britannica.com".into(), 0.85);

        // News (Medium-High Trust)
        cache.insert("bbc.com".into(), 0.80);
        cache.insert("reuters.com".into(), 0.85);
        cache.insert("apnews.com".into(), 0.85);
        cache.insert("npr.org".into(), 0.80);
        cache.insert("nytimes.com".into(), 0.80);
        cache.insert("wsj.com".into(), 0.80);
        cache.insert("economist.com".into(), 0.85);
        cache.insert("bloomberg.com".into(), 0.80);

        // Tech (Medium Trust)
        cache.insert("stackoverflow.com".into(), 0.75);
        cache.insert("github.com".into(), 0.70);
        cache.insert("docs.rs".into(), 0.80);
        cache.insert("docs.python.org".into(), 0.85);
        cache.insert("developer.mozilla.org".into(), 0.85);
        cache.insert("rust-lang.org".into(), 0.85);

        // Social / UGC (Lower Trust)
        cache.insert("reddit.com".into(), 0.40);
        cache.insert("quora.com".into(), 0.35);
        cache.insert("twitter.com".into(), 0.30);
        cache.insert("x.com".into(), 0.30);
        cache.insert("facebook.com".into(), 0.25);
        cache.insert("instagram.com".into(), 0.25);
        cache.insert("tiktok.com".into(), 0.20);
        cache.insert("youtube.com".into(), 0.40);
        cache.insert("medium.com".into(), 0.50);
        cache.insert("linkedin.com".into(), 0.50);

        Self {
            cache,
            user_overrides: HashMap::new(),
        }
    }

    /// Extract clean domain from URL.
    fn extract_domain(url: &str) -> Option<String> {
        let url = if !url.starts_with("http://") && !url.starts_with("https://") {
            format!("https://{}", url)
        } else {
            url.to_string()
        };

        // Simple URL parsing without pulling in the url crate
        let after_scheme = url.split("://").nth(1)?;
        let host = after_scheme.split('/').next()?;
        let mut domain = host.to_lowercase();
        if domain.starts_with("www.") {
            domain = domain[4..].to_string();
        }
        // Remove port
        if let Some(idx) = domain.find(':') {
            domain = domain[..idx].to_string();
        }
        Some(domain)
    }

    /// Get credibility score for a URL (0.0-1.0).
    pub fn get_score(&self, url: &str) -> f32 {
        if url.is_empty() {
            return DEFAULT_SCORE;
        }

        let domain = match Self::extract_domain(url) {
            Some(d) => d,
            None => return DEFAULT_SCORE,
        };

        // Check user overrides first
        if let Some(&score) = self.user_overrides.get(&domain) {
            return score;
        }

        // Check exact match
        if let Some(&score) = self.cache.get(&domain) {
            return score;
        }

        // Check parent domain (e.g., sub.example.com → example.com)
        let parts: Vec<&str> = domain.split('.').collect();
        if parts.len() > 2 {
            let parent = parts[parts.len() - 2..].join(".");
            if let Some(&score) = self.cache.get(&parent) {
                return score;
            }
        }

        DEFAULT_SCORE
    }

    /// Set user override for a domain.
    pub fn set_override(&mut self, domain: &str, score: f32) {
        let domain = domain.to_lowercase();
        self.user_overrides.insert(domain, score.clamp(0.0, 1.0));
    }

    /// Remove user override for a domain.
    pub fn remove_override(&mut self, domain: &str) {
        self.user_overrides.remove(&domain.to_lowercase());
    }

    /// Get all user overrides.
    pub fn get_overrides(&self) -> &HashMap<String, f32> {
        &self.user_overrides
    }
}

impl Default for SourceCredibility {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_domain() {
        let cred = SourceCredibility::new();
        assert_eq!(cred.get_score("https://arxiv.org/paper/123"), 0.90);
        assert_eq!(cred.get_score("https://reddit.com/r/rust"), 0.40);
    }

    #[test]
    fn test_unknown_domain() {
        let cred = SourceCredibility::new();
        assert_eq!(
            cred.get_score("https://random-blog.xyz/post"),
            DEFAULT_SCORE
        );
    }

    #[test]
    fn test_parent_domain_fallback() {
        let cred = SourceCredibility::new();
        // sub.reddit.com should match reddit.com
        assert_eq!(cred.get_score("https://old.reddit.com/r/rust"), 0.40);
    }

    #[test]
    fn test_www_stripping() {
        let cred = SourceCredibility::new();
        assert_eq!(cred.get_score("https://www.nature.com/articles/123"), 0.90);
    }

    #[test]
    fn test_user_override() {
        let mut cred = SourceCredibility::new();
        cred.set_override("my-company.com", 0.9);
        assert_eq!(cred.get_score("https://my-company.com/docs"), 0.9);
    }

    #[test]
    fn test_no_scheme() {
        let cred = SourceCredibility::new();
        assert_eq!(cred.get_score("nature.com/articles/123"), 0.90);
    }
}
