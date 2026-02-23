use std::collections::HashMap;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use anyhow::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use parking_lot::RwLock;

// Thread-local search buffer: zero lock overhead on hot path
thread_local! {
    static SEARCH_COUNTS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Inverted Index using Roaring Bitmaps.
pub struct InvertedIndex {
    path: PathBuf,
    bit_index: RwLock<HashMap<u16, RoaringBitmap>>,
    id_map: RwLock<HashMap<String, u32>>,
    reverse_map: RwLock<HashMap<u32, String>>,
    next_doc_id: RwLock<u32>,
}

#[derive(Serialize, Deserialize)]
struct IndexManifest {
    next_doc_id: u32,
    id_map: HashMap<String, u32>,
}

impl InvertedIndex {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            bit_index: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_map: RwLock::new(HashMap::new()),
            next_doc_id: RwLock::new(0),
        }
    }

    pub fn add(&self, external_id: &str, sdr_indices: &[u16]) {
        let mut id_map = self.id_map.write();
        let mut reverse_map = self.reverse_map.write();
        let mut next_id = self.next_doc_id.write();
        let mut bit_index = self.bit_index.write();

        let doc_id = if let Some(&id) = id_map.get(external_id) { id } else {
            let id = *next_id;
            *next_id += 1;
            id_map.insert(external_id.to_string(), id);
            reverse_map.insert(id, external_id.to_string());
            id
        };

        for &bit in sdr_indices {
            bit_index.entry(bit).or_default().insert(doc_id);
        }
    }

    pub fn add_batch(&self, documents: &[(String, Vec<u16>)]) {
        let mut id_map = self.id_map.write();
        let mut reverse_map = self.reverse_map.write();
        let mut next_id = self.next_doc_id.write();
        let mut bit_index = self.bit_index.write();

        for (external_id, sdr_indices) in documents {
            let doc_id = if let Some(&id) = id_map.get(external_id) { id } else {
                let id = *next_id;
                *next_id += 1;
                id_map.insert(external_id.clone(), id);
                reverse_map.insert(id, external_id.clone());
                id
            };
            for &bit in sdr_indices {
                bit_index.entry(bit).or_default().insert(doc_id);
            }
        }
    }

    pub fn remove(&self, external_id: &str) -> bool {
        let mut id_map = self.id_map.write();
        let mut reverse_map = self.reverse_map.write();
        if let Some(doc_id) = id_map.remove(external_id) {
            reverse_map.remove(&doc_id);
            let mut bit_index = self.bit_index.write();
            for bitmap in bit_index.values_mut() { bitmap.remove(doc_id); }
            return true;
        }
        false
    }

    pub fn search(&self, query_indices: &[u16], top_k: usize, min_overlap: u32) -> Vec<(String, u32)> {
        if query_indices.is_empty() { return vec![]; }
        let max_id = *self.next_doc_id.read() as usize;
        let bit_index = self.bit_index.read();

        let max_bits = if top_k <= 10 { 128 } else if top_k <= 50 { 256 } else { 512 };

        // Collect bitmaps and sort by rarity (smallest first = most selective)
        let mut bitmaps: Vec<&RoaringBitmap> = Vec::with_capacity(query_indices.len());
        for &bit in query_indices {
            if let Some(bm) = bit_index.get(&bit) {
                bitmaps.push(bm);
            }
        }
        if bitmaps.is_empty() { return vec![]; }
        // Only sort by rarity when we need to select a subset
        let processing_count = bitmaps.len().min(max_bits);
        if bitmaps.len() > max_bits {
            bitmaps.sort_unstable_by_key(|bm| bm.len());
        }

        // Thread-local counts buffer: zero lock overhead, sparse cleanup
        let needed = max_id + 1;

        let result = SEARCH_COUNTS.with(|cell| {
            let mut counts = cell.borrow_mut();
            if counts.len() < needed {
                counts.resize(needed, 0);
            }

            let mut active_docs: Vec<u32> = Vec::with_capacity(128);

            for bm in &bitmaps[..processing_count] {
                for doc_id in bm.iter() {
                    let idx = doc_id as usize;
                    if idx < needed {
                        if counts[idx] == 0 { active_docs.push(doc_id); }
                        counts[idx] += 1;
                    }
                }
            }

            if active_docs.is_empty() {
                return vec![];
            }

            // Filter & rank, then sparse-clean
            let mut candidates: Vec<(u32, u32)> = Vec::with_capacity(active_docs.len());
            for &doc_id in &active_docs {
                let count = counts[doc_id as usize];
                if count >= min_overlap {
                    candidates.push((doc_id, count));
                }
                counts[doc_id as usize] = 0;
            }
            candidates
        });

        if result.is_empty() { return vec![]; }
        let mut candidates = result;

        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let limit = (top_k * 10).min(500);
        if candidates.len() > limit {
            candidates.truncate(limit);
        }

        // Resolve IDs
        let reverse_map = self.reverse_map.read();
        let mut results = Vec::with_capacity(candidates.len());
        for (doc_id, count) in candidates {
            if let Some(ext_id) = reverse_map.get(&doc_id) {
                results.push((ext_id.clone(), count));
            }
        }

        results
    }

    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.path)?;
        let manifest = IndexManifest { next_doc_id: *self.next_doc_id.read(), id_map: self.id_map.read().clone() };
        serde_json::to_writer(BufWriter::new(File::create(self.path.join("index_manifest.json"))?), &manifest)?;
        let mut writer = BufWriter::new(File::create(self.path.join("sdr.idx"))?);
        let bit_index = self.bit_index.read();
        for (bit, bitmap) in bit_index.iter() {
            use byteorder::{WriteBytesExt, LittleEndian};
            writer.write_u16::<LittleEndian>(*bit)?;
            let mut buf = Vec::new();
            bitmap.serialize_into(&mut buf)?;
            writer.write_u64::<LittleEndian>(buf.len() as u64)?;
            use std::io::Write;
            writer.write_all(&buf)?;
        }
        Ok(())
    }

    pub fn load(&self) -> Result<()> {
        let manifest_path = self.path.join("index_manifest.json");
        let index_path = self.path.join("sdr.idx");
        if !manifest_path.exists() || !index_path.exists() { return Ok(()); }
        let manifest: IndexManifest = serde_json::from_reader(File::open(manifest_path)?)?;
        *self.next_doc_id.write() = manifest.next_doc_id;
        *self.id_map.write() = manifest.id_map.clone();
        
        let mut reverse = self.reverse_map.write();
        reverse.clear();
        for (k, v) in manifest.id_map.iter() {
            reverse.insert(*v, k.clone());
        }

        let mut reader = BufReader::new(File::open(index_path)?);
        let mut bit_index = self.bit_index.write();
        bit_index.clear();
        use byteorder::{ReadBytesExt, LittleEndian};
        loop {
            let bit = match reader.read_u16::<LittleEndian>() {
                Ok(b) => b,
                Err(_) => break,
            };
            let size = reader.read_u64::<LittleEndian>()?;
            let mut buf = vec![0u8; size as usize];
            use std::io::Read;
            reader.read_exact(&mut buf)?;
            bit_index.insert(bit, RoaringBitmap::deserialize_from(&buf[..])?);
        }
        Ok(())
    }

    pub fn get_stats(&self) -> (u32, usize) {
        let doc_count = *self.next_doc_id.read();
        let bit_count = self.bit_index.read().len();
        (doc_count, bit_count)
    }
}
