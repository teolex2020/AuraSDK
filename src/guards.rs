//! Store guards — dedup check, auto-protect tags, sensitive tag guard.
//!
//! Rewritten from brain_tools.py guard logic.

use std::collections::HashSet;
use regex::Regex;
use once_cell::sync::Lazy;

use crate::trust::TagTaxonomy;

// ── Regex patterns for auto-protect ──

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{2,4}")
        .expect("invalid PHONE_RE regex")
});

static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
        .expect("invalid EMAIL_RE regex")
});

static WALLET_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|T[a-zA-Z0-9]{33})")
        .expect("invalid WALLET_RE regex")
});

static API_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:sk-|api[_-]?key|token|secret)[:\s=]+[A-Za-z0-9_\-]{20,}")
        .expect("invalid API_KEY_RE regex")
});

static PASSWORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:password|passwd|пароль)[:\s=]+\S+")
        .expect("invalid PASSWORD_RE regex")
});

/// Result of running store guards.
#[derive(Debug, Clone)]
pub struct GuardResult {
    /// Additional tags to add.
    pub extra_tags: Vec<String>,
    /// Extra metadata fields to set.
    pub extra_metadata: Vec<(String, String)>,
    /// Whether human approval is needed (sensitive tag guard).
    pub needs_approval: bool,
}

/// Auto-add protective tags to content containing sensitive values.
///
/// Records with these tags are excluded from consolidation, preventing
/// data loss when LLM summarization drops specific values.
pub fn auto_protect_tags(content: &str, tags: &mut Vec<String>) {
    if PHONE_RE.is_match(content) && !tags.contains(&"contact".to_string()) {
        tags.push("contact".to_string());
    }
    if EMAIL_RE.is_match(content) && !tags.contains(&"contact".to_string()) {
        tags.push("contact".to_string());
    }
    if WALLET_RE.is_match(content) && !tags.contains(&"financial".to_string()) {
        tags.push("financial".to_string());
    }
    if (PASSWORD_RE.is_match(content) || API_KEY_RE.is_match(content))
        && !tags.contains(&"credential".to_string()) {
            tags.push("credential".to_string());
        }
}

/// Apply store guard — detect sensitive data and set actionable flag.
///
/// Interactive channels (desktop, telegram, voice) store with actionable=true
/// because the user is present. Autonomous channels store with actionable=false.
pub fn apply_store_guard(
    content: &str,
    tags: &[String],
    channel: Option<&str>,
    taxonomy: &TagTaxonomy,
) -> GuardResult {
    let is_interactive = matches!(channel, Some("desktop" | "telegram" | "voice"));
    let tag_set: HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();

    let has_sensitive_tags = taxonomy.sensitive_tags.iter().any(|t| tag_set.contains(t.as_str()));
    let has_sensitive_content = EMAIL_RE.is_match(content)
        || WALLET_RE.is_match(content)
        || API_KEY_RE.is_match(content)
        || PASSWORD_RE.is_match(content);

    if !has_sensitive_tags && !has_sensitive_content {
        return GuardResult {
            extra_tags: vec![],
            extra_metadata: vec![],
            needs_approval: false,
        };
    }

    if is_interactive {
        GuardResult {
            extra_tags: vec![],
            extra_metadata: vec![("actionable".into(), "true".into())],
            needs_approval: false,
        }
    } else {
        GuardResult {
            extra_tags: vec![],
            extra_metadata: vec![("actionable".into(), "false".into())],
            needs_approval: has_sensitive_tags,
        }
    }
}

/// Check if content should be skipped from consolidation based on its tags.
pub fn should_skip_consolidation(tags: &[String], taxonomy: &TagTaxonomy) -> bool {
    let tag_set: HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();
    taxonomy.consolidation_skip_tags.iter().any(|t| tag_set.contains(t.as_str()))
}

/// Check if a record is protected from archival.
pub fn is_archive_protected(tags: &[String], taxonomy: &TagTaxonomy) -> bool {
    let tag_set: HashSet<&str> = tags.iter().map(|s| s.as_str()).collect();
    taxonomy.archive_protected_tags.iter().any(|t| tag_set.contains(t.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_protect_phone() {
        let mut tags = vec!["personal".to_string()];
        auto_protect_tags("My phone is +380991234567", &mut tags);
        assert!(tags.contains(&"contact".to_string()));
    }

    #[test]
    fn test_auto_protect_email() {
        let mut tags = vec![];
        auto_protect_tags("Email: test@example.com", &mut tags);
        assert!(tags.contains(&"contact".to_string()));
    }

    #[test]
    fn test_auto_protect_wallet() {
        let mut tags = vec![];
        auto_protect_tags("Send to 0x1234567890abcdef1234567890abcdef12345678", &mut tags);
        assert!(tags.contains(&"financial".to_string()));
    }

    #[test]
    fn test_auto_protect_api_key() {
        let mut tags = vec![];
        auto_protect_tags("api_key: sk-proj-abc123def456ghi789jkl012mno", &mut tags);
        assert!(tags.contains(&"credential".to_string()));
    }

    #[test]
    fn test_store_guard_sensitive_autonomous() {
        let taxonomy = TagTaxonomy::default();
        let tags = vec!["financial".to_string()];
        let result = apply_store_guard("wallet info", &tags, Some("agent"), &taxonomy);
        assert_eq!(result.extra_metadata[0].1, "false"); // actionable = false
        assert!(result.needs_approval);
    }

    #[test]
    fn test_store_guard_sensitive_interactive() {
        let taxonomy = TagTaxonomy::default();
        let tags = vec!["financial".to_string()];
        let result = apply_store_guard("wallet info", &tags, Some("telegram"), &taxonomy);
        assert_eq!(result.extra_metadata[0].1, "true"); // actionable = true
        assert!(!result.needs_approval);
    }

    #[test]
    fn test_should_skip_consolidation() {
        let taxonomy = TagTaxonomy::default();
        assert!(should_skip_consolidation(&["credential".into()], &taxonomy));
        assert!(!should_skip_consolidation(&["random".into()], &taxonomy));
    }
}
