#[cfg(feature = "python")]
use pyo3::prelude::*;

// SDR resolution based on feature flags
#[cfg(feature = "lite")]
const DEFAULT_TOTAL_BITS: usize = 16384;  // 16k bits for embedded
#[cfg(feature = "lite")]
const DEFAULT_NUM_ACTIVE: usize = 128;
#[cfg(feature = "lite")]
const DEFAULT_PROTECTED_RANGE: (usize, usize) = (0, 1024);
#[cfg(feature = "lite")]
const DEFAULT_GENERAL_RANGE: (usize, usize) = (1024, 16384);

#[cfg(not(feature = "lite"))]
const DEFAULT_TOTAL_BITS: usize = 262144;  // 256k bits (High Sparsity)
#[cfg(not(feature = "lite"))]
const DEFAULT_NUM_ACTIVE: usize = 512;
#[cfg(not(feature = "lite"))]
const DEFAULT_PROTECTED_RANGE: (usize, usize) = (0, 4096);
#[cfg(not(feature = "lite"))]
const DEFAULT_GENERAL_RANGE: (usize, usize) = (4096, 262144);

/// SDR Interpreter - Sparse Distributed Representation for semantic encoding.
///
/// Resolution modes:
/// - Full (default): 256k bits, 512 active - maximum precision
/// - Lite (feature: lite): 16k bits, 128 active - for embedded devices
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyclass)]
pub struct SDRInterpreter {
    pub total_bits: usize,
    pub num_active: usize,
    pub protected_range: (usize, usize),
    pub general_range: (usize, usize),
}

impl Default for SDRInterpreter {
    fn default() -> Self {
        Self {
            total_bits: DEFAULT_TOTAL_BITS,
            num_active: DEFAULT_NUM_ACTIVE,
            protected_range: DEFAULT_PROTECTED_RANGE,
            general_range: DEFAULT_GENERAL_RANGE,
        }
    }
}

impl SDRInterpreter {
    /// Create a lite-mode SDR interpreter (16k bits) for embedded devices
    pub fn lite() -> Self {
        Self {
            total_bits: 16384,
            num_active: 128,
            protected_range: (0, 1024),
            general_range: (1024, 16384),
        }
    }

    /// Create a custom resolution SDR interpreter
    pub fn with_resolution(total_bits: usize, num_active: usize) -> Self {
        let protected_size = total_bits / 64;
        Self {
            total_bits,
            num_active,
            protected_range: (0, protected_size),
            general_range: (protected_size, total_bits),
        }
    }
}

impl SDRInterpreter {
    /// Internal optimized Tanimoto for Rust usage (slices)
    #[inline]
    pub fn tanimoto_sparse(&self, a: &[u16], b: &[u16]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Intersection of two sorted arrays
        let mut intersection = 0;
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                intersection += 1;
                i += 1;
                j += 1;
            }
        }

        let union = a.len() + b.len() - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Convert text to SDR (internal Rust version).
    /// Optimized: uses Vec+sort+dedup instead of HashSet for zero-alloc hot path.
    pub fn text_to_sdr(&self, text: &str, is_identity: bool) -> Vec<u16> {
        self.text_to_sdr_inner(text, is_identity, false)
    }

    /// Fast path: skip to_lowercase() when caller guarantees input is already lowered.
    #[inline]
    pub fn text_to_sdr_lowered(&self, text: &str, is_identity: bool) -> Vec<u16> {
        self.text_to_sdr_inner(text, is_identity, true)
    }

    fn text_to_sdr_inner(&self, text: &str, is_identity: bool, pre_lowered: bool) -> Vec<u16> {
        let bit_range = if is_identity {
            self.protected_range
        } else {
            self.general_range
        };
        let range_size = bit_range.1 - bit_range.0;
        let base = bit_range.0;

        // ASCII fast path: avoid String allocation + Vec<char> allocation
        let trimmed = text.trim();
        let is_ascii = trimmed.is_ascii();

        if is_ascii {
            // Zero-alloc path for ASCII text (RF-SIG, TGT, frequencies, etc.)
            let bytes = trimmed.as_bytes();
            let len = bytes.len();
            if len == 0 { return vec![]; }

            let estimated = if len >= 4 { (len - 3) * 20 + (len - 2) * 2 + (len - 1) } else { len * 3 };
            let mut indices: Vec<u16> = Vec::with_capacity(estimated);
            let mut gram_buf = [0u8; 4];

            // --- 1. Quadgrams (90% signal, 20 bits per gram) ---
            if len >= 4 {
                for i in 0..=(len - 4) {
                    let mut has_digit = false;
                    for j in 0..4 {
                        let b = bytes[i + j];
                        gram_buf[j] = if pre_lowered { b } else { b.to_ascii_lowercase() };
                        if b.is_ascii_digit() { has_digit = true; }
                    }

                    let mut seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..4]);
                    if has_digit {
                        seed = (seed ^ 0x1234567812345678).rotate_right(3);
                    }

                    for k in 0..20u64 {
                        let idx = (seed.wrapping_add(k.wrapping_mul(9999))) as usize % range_size;
                        indices.push((base + idx) as u16);
                    }
                }
            }

            // --- 2. Trigrams (5% signal, 2 bits per gram) ---
            if len >= 3 {
                for i in 0..=(len - 3) {
                    let mut has_digit = false;
                    for j in 0..3 {
                        let b = bytes[i + j];
                        gram_buf[j] = if pre_lowered { b } else { b.to_ascii_lowercase() };
                        if b.is_ascii_digit() { has_digit = true; }
                    }

                    let mut seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..3]);
                    if has_digit {
                        seed = (seed ^ 0x5F3759DF5F3759DF).rotate_left(7);
                    }

                    for k in 0..2u64 {
                        let idx = (seed.wrapping_add(k.wrapping_mul(1337))) as usize % range_size;
                        indices.push((base + idx) as u16);
                    }
                }
            } else {
                // Very short text
                let mut lowered_buf = [0u8; 4];
                for (j, &b) in bytes.iter().enumerate().take(4) {
                    lowered_buf[j] = if pre_lowered { b } else { b.to_ascii_lowercase() };
                }
                let seed = xxhash_rust::xxh3::xxh3_64(&lowered_buf[..bytes.len()]);
                for k in 0..2u64 {
                    let idx = (seed.wrapping_add(k.wrapping_mul(1337))) as usize % range_size;
                    indices.push((base + idx) as u16);
                }
            }

            // --- 3. Bigrams (5% signal, 1 bit per gram) ---
            if len >= 2 {
                for i in 0..=(len - 2) {
                    gram_buf[0] = if pre_lowered { bytes[i] } else { bytes[i].to_ascii_lowercase() };
                    gram_buf[1] = if pre_lowered { bytes[i + 1] } else { bytes[i + 1].to_ascii_lowercase() };
                    let seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..2]);
                    let idx = seed as usize % range_size;
                    indices.push((base + idx) as u16);
                }
            }

            indices.sort_unstable();
            indices.dedup();
            return indices;
        }

        // UTF-8 fallback (non-ASCII text: Cyrillic, CJK, emoji, etc.)
        let text_lower = if pre_lowered {
            trimmed.to_string()
        } else {
            trimmed.to_lowercase()
        };
        let chars: Vec<char> = text_lower.chars().collect();
        let len = chars.len();
        if len == 0 {
            return vec![];
        }

        let estimated = if len >= 4 { (len - 3) * 20 + (len - 2) * 2 + (len - 1) } else { len * 3 };
        let mut indices: Vec<u16> = Vec::with_capacity(estimated);
        let mut gram_buf = [0u8; 16];

        // --- 1. Quadgrams ---
        if len >= 4 {
            for w in chars.windows(4) {
                let mut pos = 0;
                let mut has_digit = false;
                for &c in w {
                    let encoded = c.encode_utf8(&mut gram_buf[pos..]);
                    pos += encoded.len();
                    if c.is_ascii_digit() { has_digit = true; }
                }

                let mut seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..pos]);
                if has_digit {
                    seed = (seed ^ 0x1234567812345678).rotate_right(3);
                }

                for k in 0..20u64 {
                    let idx = (seed.wrapping_add(k.wrapping_mul(9999))) as usize % range_size;
                    indices.push((base + idx) as u16);
                }
            }
        }

        // --- 2. Trigrams ---
        if len >= 3 {
            for w in chars.windows(3) {
                let mut pos = 0;
                let mut has_digit = false;
                for &c in w {
                    let encoded = c.encode_utf8(&mut gram_buf[pos..]);
                    pos += encoded.len();
                    if c.is_ascii_digit() { has_digit = true; }
                }

                let mut seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..pos]);
                if has_digit {
                    seed = (seed ^ 0x5F3759DF5F3759DF).rotate_left(7);
                }

                for k in 0..2u64 {
                    let idx = (seed.wrapping_add(k.wrapping_mul(1337))) as usize % range_size;
                    indices.push((base + idx) as u16);
                }
            }
        } else {
            let seed = xxhash_rust::xxh3::xxh3_64(text_lower.as_bytes());
            for k in 0..2u64 {
                let idx = (seed.wrapping_add(k.wrapping_mul(1337))) as usize % range_size;
                indices.push((base + idx) as u16);
            }
        }

        // --- 3. Bigrams ---
        if len >= 2 {
            for w in chars.windows(2) {
                let mut pos = 0;
                for &c in w {
                    let encoded = c.encode_utf8(&mut gram_buf[pos..]);
                    pos += encoded.len();
                }
                let seed = xxhash_rust::xxh3::xxh3_64(&gram_buf[..pos]);
                let idx = seed as usize % range_size;
                indices.push((base + idx) as u16);
            }
        }

        indices.sort_unstable();
        indices.dedup();
        indices
    }
    
    // Helper used in tests
    #[cfg(test)]
    pub fn to_dense(&self, sparse: &Vec<u16>) -> Vec<u8> {
        let mut dense = vec![0u8; self.total_bits];
        for &idx in sparse {
            if idx < self.total_bits as u16 {
                dense[idx as usize] = 1;
            }
        }
        dense
    }

    #[cfg(test)]
    pub fn tanimoto_dense(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut intersection = 0;
        let mut union = 0;
        for i in 0..self.total_bits {
            if a[i] == 1 && b[i] == 1 {
                intersection += 1;
            }
            if a[i] == 1 || b[i] == 1 {
                union += 1;
            }
        }
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }
}

// ============= PYTHON BINDINGS =============
#[cfg(feature = "python")]
#[allow(unused_imports, non_local_definitions)]
mod python_sdr {
    use super::*;
    use pyo3::prelude::*;

    #[pymethods]
    impl SDRInterpreter {
        #[new]
        #[pyo3(signature = (total_bits=262144, num_active=512))]
        pub fn new(total_bits: usize, num_active: usize) -> Self {
            Self {
                total_bits,
                num_active,
                protected_range: (0, 4096),
                general_range: (4096, total_bits),
            }
        }

        /// Convert text to SDR (returns list of active indices).
        #[pyo3(name = "text_to_sdr", signature = (text, is_identity=false))]
        pub fn text_to_sdr_py(&self, text: &str, is_identity: bool) -> Vec<u16> {
            self.text_to_sdr(text, is_identity)
        }

        /// Batch process multiple strings into a list of SDR lists.
        pub fn batch_text_to_sdr(
            &self,
            texts: Vec<String>,
            is_identity: bool
        ) -> Vec<Vec<u16>> {
            texts.iter().map(|t| self.text_to_sdr(t, is_identity)).collect()
        }

        /// Calculate Tanimoto similarity (Python wrapper).
        #[pyo3(name = "tanimoto_sparse")]
        pub fn tanimoto_sparse_py(&self, a: Vec<u16>, b: Vec<u16>) -> f32 {
            self.tanimoto_sparse(&a, &b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_generation() {
        let sdr = SDRInterpreter::default();
        let indices1 = sdr.text_to_sdr("Hello World", false);
        let indices2 = sdr.text_to_sdr("Hello World", false);
        
        assert_eq!(indices1, indices2, "SDR must be deterministic");
        assert!(!indices1.is_empty());
    }

    #[test]
    fn test_similarity() {
        let sdr = SDRInterpreter::default();
        let s1 = sdr.text_to_sdr("Apple", false);
        let s2 = sdr.text_to_sdr("Apple Pie", false); // Overlap
        let s3 = sdr.text_to_sdr("Banana", false); // Different

        let sim1 = sdr.tanimoto_sparse(&s1, &s2);
        let sim2 = sdr.tanimoto_sparse(&s1, &s3);

        assert!(sim1 > sim2, "Similar texts should have higher score");
    }
    
    #[test]
    fn test_sdr_deterministic() {
        // Verify SDR generation is deterministic
        let sdr = SDRInterpreter::default();
        let s1 = sdr.text_to_sdr("Rust is fast", false);
        let s2 = sdr.text_to_sdr("Rust is fast", false);

        assert_eq!(s1, s2, "Same input should produce identical SDR");
        assert!(!s1.is_empty(), "SDR should not be empty");
    }

    #[test]
    fn test_paraphrase_tanimoto_range() {
        // Verify SDR Tanimoto ranges for claim grouping calibration.
        // Same-topic paraphrases should score > 0.15 (CLAIM_SIMILARITY_THRESHOLD),
        // while cross-topic pairs should score < 0.15.
        let sdr = SDRInterpreter::default();

        // Same-topic pair (near-identical)
        let sa = sdr.text_to_sdr("deploy to staging before production release", false);
        let sb = sdr.text_to_sdr("deploy to staging before production release always", false);
        let sim_identical = sdr.tanimoto_sparse(&sa, &sb);
        assert!(sim_identical > 0.15, "near-identical: {:.3}", sim_identical);

        // Same-topic pair (moderate paraphrase)
        let sa = sdr.text_to_sdr("Always run integration tests before merging pull requests", false);
        let sb = sdr.text_to_sdr("Run all integration tests before merging any pull request", false);
        let sim_paraphrase = sdr.tanimoto_sparse(&sa, &sb);
        assert!(sim_paraphrase > 0.15, "moderate paraphrase: {:.3}", sim_paraphrase);

        // Cross-topic pair (should be low)
        let sa = sdr.text_to_sdr("Configure blue-green deployment pipeline for zero-downtime releases", false);
        let sb = sdr.text_to_sdr("Configure PostgreSQL connection pool with maximum twenty-five connections", false);
        let sim_cross = sdr.tanimoto_sparse(&sa, &sb);
        assert!(sim_cross < 0.15, "cross-topic: {:.3}", sim_cross);

    }
}
