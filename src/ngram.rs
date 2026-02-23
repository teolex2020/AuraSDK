//! MinHash N-gram index for fuzzy text matching.
//!
//! Rewritten from aura-cognitive ngram.py.
//! Replaces numpy vectorization with pure Rust — target: <100ms for 537 records.

use ahash::AHashMap;

use crate::synonym::SynonymRing;

const PRIME: u64 = 2_147_483_647; // Mersenne prime 2^31-1
const DEFAULT_NUM_HASHES: usize = 128;

/// MinHash-based n-gram index for approximate Jaccard similarity.
pub struct NGramIndex {
    num_hashes: usize,
    /// Coefficients for hash functions: h(x) = (a*x + b) % PRIME
    a: Vec<u64>,
    b: Vec<u64>,
    /// Record ID → MinHash signature (num_hashes elements).
    signatures: AHashMap<String, Vec<u64>>,
    /// LSH buckets: for each hash position, hash_value → set of record IDs.
    buckets: Vec<AHashMap<u64, Vec<String>>>,
    /// Optional synonym ring for query expansion.
    synonym_ring: Option<SynonymRing>,
}

impl NGramIndex {
    /// Create a new n-gram index.
    pub fn new(num_hashes: Option<usize>, synonym_ring: Option<SynonymRing>) -> Self {
        let num_hashes = num_hashes.unwrap_or(DEFAULT_NUM_HASHES);

        // Generate random hash function coefficients
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a: Vec<u64> = (0..num_hashes).map(|_| rng.gen_range(1..PRIME)).collect();
        let b: Vec<u64> = (0..num_hashes).map(|_| rng.gen_range(0..PRIME)).collect();

        let buckets = (0..num_hashes).map(|_| AHashMap::new()).collect();

        Self {
            num_hashes,
            a,
            b,
            signatures: AHashMap::new(),
            buckets,
            synonym_ring,
        }
    }

    /// Tokenize text into character trigram hashes.
    fn tokenize(text: &str) -> Vec<u64> {
        let clean: String = text
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        let clean = clean.split_whitespace().collect::<Vec<_>>().join(" ");

        if clean.len() < 3 {
            if clean.len() == 2 {
                return vec![Self::hash_str(&clean)];
            } else if clean.len() == 1 {
                return vec![Self::hash_str(&clean)];
            }
            return vec![];
        }

        let bytes = clean.as_bytes();
        let mut shingles = Vec::with_capacity(bytes.len() - 2);
        for i in 0..=(bytes.len() - 3) {
            let trigram = &bytes[i..i + 3];
            let h = xxhash_rust::xxh3::xxh3_64(trigram) & 0x7FFFFFFF;
            shingles.push(h);
        }

        shingles.sort_unstable();
        shingles.dedup();
        shingles
    }

    fn hash_str(s: &str) -> u64 {
        xxhash_rust::xxh3::xxh3_64(s.as_bytes()) & 0x7FFFFFFF
    }

    /// Expand text using synonym ring.
    fn expand(&self, text: &str) -> String {
        if let Some(ref ring) = self.synonym_ring {
            ring.expand(text)
        } else {
            text.to_string()
        }
    }

    /// Compute MinHash signature for a set of shingles.
    fn minhash(&self, shingles: &[u64]) -> Vec<u64> {
        let mut sig = vec![u64::MAX; self.num_hashes];

        for &s in shingles {
            for i in 0..self.num_hashes {
                let h = (self.a[i].wrapping_mul(s).wrapping_add(self.b[i])) % PRIME;
                if h < sig[i] {
                    sig[i] = h;
                }
            }
        }

        sig
    }

    /// Index a record.
    pub fn add(&mut self, record_id: &str, text: &str) {
        let expanded = self.expand(text);
        let shingles = Self::tokenize(&expanded);
        if shingles.is_empty() {
            return;
        }

        let sig = self.minhash(&shingles);

        // Populate LSH buckets
        for (i, &val) in sig.iter().enumerate() {
            self.buckets[i]
                .entry(val)
                .or_default()
                .push(record_id.to_string());
        }

        self.signatures.insert(record_id.to_string(), sig);
    }

    /// Remove a record from the index.
    pub fn remove(&mut self, record_id: &str) {
        if let Some(sig) = self.signatures.remove(record_id) {
            for (i, &val) in sig.iter().enumerate() {
                if let Some(bucket) = self.buckets[i].get_mut(&val) {
                    bucket.retain(|id| id != record_id);
                }
            }
        }
    }

    /// Query for similar records using LSH.
    ///
    /// Returns: [(similarity, record_id), ...] sorted descending.
    pub fn query(&self, text: &str, top_k: usize) -> Vec<(f32, String)> {
        let expanded = self.expand(text);
        let shingles = Self::tokenize(&expanded);
        if shingles.is_empty() {
            return vec![];
        }

        let query_sig = self.minhash(&shingles);

        // Count matching hash components per candidate
        let mut candidates: AHashMap<&str, u32> = AHashMap::new();
        for (i, &val) in query_sig.iter().enumerate() {
            if let Some(bucket) = self.buckets[i].get(&val) {
                for rid in bucket {
                    *candidates.entry(rid.as_str()).or_insert(0) += 1;
                }
            }
        }

        // Convert to similarity estimates
        let mut results: Vec<(f32, String)> = candidates
            .into_iter()
            .map(|(rid, count)| {
                let similarity = count as f32 / self.num_hashes as f32;
                (similarity, rid.to_string())
            })
            .filter(|(sim, _)| *sim > 0.0)
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Compute exact Jaccard similarity between two record signatures.
    pub fn jaccard(&self, id_a: &str, id_b: &str) -> f32 {
        let sig_a = match self.signatures.get(id_a) {
            Some(s) => s,
            None => return 0.0,
        };
        let sig_b = match self.signatures.get(id_b) {
            Some(s) => s,
            None => return 0.0,
        };

        let matching = sig_a.iter().zip(sig_b.iter()).filter(|(a, b)| a == b).count();
        matching as f32 / self.num_hashes as f32
    }

    /// Find all pairs with Jaccard similarity >= threshold.
    pub fn find_similar_pairs(&self, threshold: f32) -> Vec<(String, String, f32)> {
        let ids: Vec<&String> = self.signatures.keys().collect();
        let mut pairs = Vec::new();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let sim = self.jaccard(ids[i], ids[j]);
                if sim >= threshold {
                    pairs.push((ids[i].clone(), ids[j].clone(), sim));
                }
            }
        }

        pairs
    }

    /// Number of indexed records.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Check if a record is indexed.
    pub fn contains(&self, record_id: &str) -> bool {
        self.signatures.contains_key(record_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_query() {
        let mut idx = NGramIndex::new(None, None);
        idx.add("r1", "The quick brown fox jumps over the lazy dog");
        idx.add("r2", "The quick brown fox");
        idx.add("r3", "Something completely different");

        let results = idx.query("quick brown fox", 10);
        assert!(!results.is_empty());
        // r1 and r2 should rank higher than r3
        let r1_pos = results.iter().position(|(_, id)| id == "r1");
        let r3_pos = results.iter().position(|(_, id)| id == "r3");
        if let (Some(p1), Some(p3)) = (r1_pos, r3_pos) {
            assert!(p1 < p3);
        }
    }

    #[test]
    fn test_jaccard() {
        let mut idx = NGramIndex::new(None, None);
        idx.add("a", "hello world foo bar");
        idx.add("b", "hello world foo bar");
        idx.add("c", "completely different text here");

        let sim_ab = idx.jaccard("a", "b");
        let sim_ac = idx.jaccard("a", "c");
        assert!(sim_ab > sim_ac, "Identical texts should have higher Jaccard");
        assert!((sim_ab - 1.0).abs() < 0.01, "Identical texts should be ~1.0");
    }

    #[test]
    fn test_remove() {
        let mut idx = NGramIndex::new(None, None);
        idx.add("r1", "test content");
        assert_eq!(idx.len(), 1);
        idx.remove("r1");
        assert_eq!(idx.len(), 0);
    }
}
