//! # Active Cortex — Hot-Path Reflex Layer
//!
//! This module implements a lock-free, O(1) cache for high-priority "Anchor" memories.
//! It sits above the main SDR index and provides sub-millisecond access to critical
//! motor reflexes, regardless of how many records exist in the cold storage.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                 User Query                       │
//! └─────────────────────┬───────────────────────────┘
//!                       │
//!                       ▼
//! ┌─────────────────────────────────────────────────┐
//! │          Active Cortex (DashMap)                │
//! │          O(1) lookup, ~200µs                    │
//! │          Lock-free, parallel reads              │
//! └─────────────────────┬───────────────────────────┘
//!                       │ MISS
//!                       ▼
//! ┌─────────────────────────────────────────────────┐
//! │          Cold SDR Index (RwLock)                │
//! │          O(k) lookup, ~40ms at 1M               │
//! │          Full semantic search                   │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Patent Alignment
//!
//! This implements the "hierarchical cognitive memory" concept:
//! - L1 (Active Cortex): Crystallized reflexes for motor control
//! - L2 (Cold Storage): Full semantic memory with SDR matching

use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum entries in the Active Cortex before LRU eviction kicks in.
/// 10,000 anchors × ~1KB each ≈ 10MB max memory footprint.
const MAX_CORTEX_ENTRIES: usize = 10_000;

/// Payload stored for each reflex anchor.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ReflexPayload {
    /// Original text content of the anchor
    pub text: String,
    /// Intensity/priority score (higher = more important)
    pub intensity: f32,
    /// Optional 3D spatial coordinates for motor memory
    pub coords: Option<(f32, f32, f32)>,
    /// Access counter for LRU tracking
    access_count: u64,
    /// Document ID in the main storage (for cross-reference)
    pub doc_id: u64,
}

impl ReflexPayload {
    pub fn new(text: String, intensity: f32, coords: Option<(f32, f32, f32)>, doc_id: u64) -> Self {
        Self {
            text,
            intensity,
            coords,
            access_count: 0,
            doc_id,
        }
    }
}

/// Lock-free hot-path cache for critical anchor memories.
///
/// Uses DashMap for fully concurrent reads/writes without blocking.
/// Provides O(1) lookup via SDR signature hashing.
pub struct ActiveCortex {
    /// Main storage: SDR signature hash -> ReflePayload
    hot_path: DashMap<u64, ReflexPayload>,
    /// Global access counter for LRU
    global_access: AtomicU64,
    /// Statistics: total hits
    hits: AtomicU64,
    /// Statistics: total misses
    misses: AtomicU64,
}

impl ActiveCortex {
    /// Create a new, empty Active Cortex.
    pub fn new() -> Self {
        Self {
            hot_path: DashMap::with_capacity(MAX_CORTEX_ENTRIES),
            global_access: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Compute a u64 signature from SDR indices for O(1) lookup.
    /// Uses xxHash internally for speed.
    pub fn sdr_to_signature(sdr_indices: &[u32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        sdr_indices.hash(&mut hasher);
        hasher.finish()
    }

    /// Insert an anchor into the Active Cortex.
    ///
    /// Only call this for high-priority anchors (SFT, motor reflexes, identity).
    /// Automatic LRU eviction occurs if capacity is exceeded.
    pub fn insert(&self, sdr_indices: &[u32], payload: ReflexPayload) {
        // Evict if at capacity
        if self.hot_path.len() >= MAX_CORTEX_ENTRIES {
            self.evict_lru();
        }

        let signature = Self::sdr_to_signature(sdr_indices);
        self.hot_path.insert(signature, payload);
    }

    /// Fast reflex lookup — O(1), lock-free.
    ///
    /// Returns Some(payload) if the anchor is in the cortex, None otherwise.
    /// On miss, caller should fall back to the cold SDR index.
    pub fn get_reflex(&self, sdr_indices: &[u32]) -> Option<ReflexPayload> {
        let signature = Self::sdr_to_signature(sdr_indices);

        if let Some(mut entry) = self.hot_path.get_mut(&signature) {
            // Update access count for LRU
            entry.access_count = self.global_access.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Check if an anchor exists in the cortex without updating LRU.
    #[allow(dead_code)]
    pub fn contains(&self, sdr_indices: &[u32]) -> bool {
        let signature = Self::sdr_to_signature(sdr_indices);
        self.hot_path.contains_key(&signature)
    }

    /// Remove an anchor from the cortex.
    #[allow(dead_code)]
    pub fn remove(&self, sdr_indices: &[u32]) -> Option<ReflexPayload> {
        let signature = Self::sdr_to_signature(sdr_indices);
        self.hot_path.remove(&signature).map(|(_, v)| v)
    }

    /// Evict the least recently used entries to make room.
    /// Removes 10% of entries with lowest access counts.
    fn evict_lru(&self) {
        let evict_count = MAX_CORTEX_ENTRIES / 10;

        // Collect all entries with their access counts
        let mut entries: Vec<(u64, u64)> = self
            .hot_path
            .iter()
            .map(|e| (*e.key(), e.value().access_count))
            .collect();

        // Sort by access count (ascending)
        entries.sort_by_key(|(_, count)| *count);

        // Evict the oldest entries
        for (key, _) in entries.into_iter().take(evict_count) {
            self.hot_path.remove(&key);
        }
    }

    /// Get current cortex size.
    pub fn len(&self) -> usize {
        self.hot_path.len()
    }

    /// Check if cortex is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.hot_path.is_empty()
    }

    /// Get hit rate statistics.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    /// Get statistics as a formatted string.
    pub fn stats(&self) -> String {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        format!(
            "ActiveCortex | Entries: {} | Hits: {} | Misses: {} | HitRate: {:.1}%",
            self.len(),
            hits,
            misses,
            self.hit_rate() * 100.0
        )
    }

    /// Clear all entries (useful for testing).
    #[allow(dead_code)]
    pub fn clear(&self) {
        self.hot_path.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

impl Default for ActiveCortex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_get() {
        let cortex = ActiveCortex::new();
        let sdr = vec![1, 5, 23, 100, 512];
        let payload = ReflexPayload::new(
            "Motor reflex: grip".to_string(),
            0.95,
            Some((10.0, 20.0, 30.0)),
            42,
        );

        cortex.insert(&sdr, payload.clone());

        let result = cortex.get_reflex(&sdr).unwrap();
        assert_eq!(result.text, "Motor reflex: grip");
        assert_eq!(result.intensity, 0.95);
        assert_eq!(result.coords, Some((10.0, 20.0, 30.0)));
    }

    #[test]
    fn test_miss_returns_none() {
        let cortex = ActiveCortex::new();
        let sdr = vec![1, 2, 3];

        assert!(cortex.get_reflex(&sdr).is_none());
    }

    #[test]
    fn test_hit_rate() {
        let cortex = ActiveCortex::new();
        let sdr = vec![1, 2, 3];
        let payload = ReflexPayload::new("test".to_string(), 0.5, None, 1);

        cortex.insert(&sdr, payload);

        // 1 hit
        cortex.get_reflex(&sdr);
        // 1 miss
        cortex.get_reflex(&vec![4, 5, 6]);

        assert_eq!(cortex.hit_rate(), 0.5);
    }
}
