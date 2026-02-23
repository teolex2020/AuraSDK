//! Two-tier caching — in-memory recall cache + persistent search cache.
//!
//! Rewritten from brain_tools.py cache logic.

use std::collections::HashMap;
use std::time::Instant;
use parking_lot::Mutex;

/// In-memory recall cache entry.
struct CacheEntry {
    result: String,
    inserted_at: Instant,
}

/// Two-tier cache for recall and search results.
pub struct RecallCache {
    /// In-memory cache: normalized_query → (result, timestamp).
    entries: Mutex<HashMap<String, CacheEntry>>,
    /// TTL in seconds.
    ttl_secs: u64,
    /// Maximum entries before eviction.
    max_entries: usize,
}

impl RecallCache {
    pub fn new(ttl_secs: u64, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            ttl_secs,
            max_entries,
        }
    }

    /// Normalize query key for cache lookup.
    fn normalize(query: &str) -> String {
        query.to_lowercase().trim().to_string()
    }

    /// Get cached recall result, or None if miss/expired.
    pub fn get(&self, query: &str) -> Option<String> {
        let key = Self::normalize(query);
        let mut entries = self.entries.lock();

        if let Some(entry) = entries.get(&key) {
            if entry.inserted_at.elapsed().as_secs() < self.ttl_secs {
                return Some(entry.result.clone());
            }
            // Expired
            entries.remove(&key);
        }
        None
    }

    /// Store recall result in cache.
    pub fn put(&self, query: &str, result: String) {
        let key = Self::normalize(query);
        let mut entries = self.entries.lock();

        // Evict oldest if at capacity
        if entries.len() >= self.max_entries {
            let oldest_key = entries
                .iter()
                .min_by_key(|(_, v)| v.inserted_at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                entries.remove(&k);
            }
        }

        entries.insert(key, CacheEntry {
            result,
            inserted_at: Instant::now(),
        });
    }

    /// Clear the entire cache. Called on store/update/delete to prevent stale data.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }

    /// Remove expired entries.
    pub fn evict_expired(&self) {
        let mut entries = self.entries.lock();
        entries.retain(|_, entry| entry.inserted_at.elapsed().as_secs() < self.ttl_secs);
    }
}

impl Default for RecallCache {
    fn default() -> Self {
        Self::new(300, 50) // 5 min TTL, max 50 entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cache_hit() {
        let cache = RecallCache::new(60, 10);
        cache.put("Who is Teo?", "Teo loves Rust".into());
        assert_eq!(cache.get("Who is Teo?"), Some("Teo loves Rust".into()));
        assert_eq!(cache.get("who is teo?"), Some("Teo loves Rust".into())); // case insensitive
    }

    #[test]
    fn test_cache_miss() {
        let cache = RecallCache::new(60, 10);
        assert_eq!(cache.get("unknown query"), None);
    }

    #[test]
    fn test_cache_expiry() {
        let cache = RecallCache::new(1, 10); // 1 second TTL
        cache.put("test", "result".into());
        assert!(cache.get("test").is_some());
        thread::sleep(Duration::from_secs(2));
        assert!(cache.get("test").is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = RecallCache::new(60, 10);
        cache.put("a", "1".into());
        cache.put("b", "2".into());
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = RecallCache::new(60, 2); // max 2 entries
        cache.put("a", "1".into());
        cache.put("b", "2".into());
        cache.put("c", "3".into()); // should evict oldest
        assert_eq!(cache.len(), 2);
    }
}
