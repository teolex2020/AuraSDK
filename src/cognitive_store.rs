//! Cognitive binary storage — append-only log with snapshots.
//!
//! Rewritten from aura-cognitive store.py.
//! Format: COG1 header → [OP(1B) | payload_len(4B) | CRC32(4B) | payload]...

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::record::Record;

const MAGIC: &[u8; 4] = b"COG1";
const VERSION: u8 = 2;

const OP_STORE: u8 = 0x01;
const OP_UPDATE: u8 = 0x02;
const OP_DELETE: u8 = 0x03;

const SNAP_MAGIC: &[u8; 4] = b"CSN1";

/// Append-only cognitive record storage with snapshot-accelerated loading.
pub struct CognitiveStore {
    #[allow(dead_code)]
    path: PathBuf,
    log_path: PathBuf,
    snap_path: PathBuf,
    writer: Mutex<Option<BufWriter<File>>>,
    log_position: Mutex<u64>,
}

impl CognitiveStore {
    /// Open or create a cognitive store at the given directory.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let log_path = path.join("brain.cog");
        let snap_path = path.join("brain.snap");

        // Initialize log file if it doesn't exist
        if !log_path.exists() {
            let mut f = File::create(&log_path)?;
            f.write_all(MAGIC)?;
            f.write_u8(VERSION)?;
            f.flush()?;
        }

        let writer_file = OpenOptions::new().append(true).open(&log_path)?;
        let writer = BufWriter::new(writer_file);

        Ok(Self {
            path,
            log_path,
            snap_path,
            writer: Mutex::new(Some(writer)),
            log_position: Mutex::new(0),
        })
    }

    /// Load all records from snapshot + log replay.
    pub fn load_all(&self) -> Result<HashMap<String, Record>> {
        let mut records = HashMap::new();

        // 1. Load snapshot if exists
        let snap_end_pos = if self.snap_path.exists() {
            self.load_snapshot(&mut records)?
        } else {
            5 // Skip magic(4) + version(1)
        };

        // 2. Replay log entries after snapshot position
        self.replay_log(&mut records, snap_end_pos)?;

        *self.log_position.lock() = snap_end_pos;

        Ok(records)
    }

    /// Load records from snapshot file.
    fn load_snapshot(&self, records: &mut HashMap<String, Record>) -> Result<u64> {
        let mut reader = BufReader::new(File::open(&self.snap_path)?);

        // Verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != SNAP_MAGIC {
            return Err(anyhow!("Invalid snapshot magic"));
        }

        let _version = reader.read_u8()?;
        let log_position = reader.read_u64::<LittleEndian>()?;
        let record_count = reader.read_u32::<LittleEndian>()?;

        for _ in 0..record_count {
            let payload_len = reader.read_u32::<LittleEndian>()? as usize;
            let mut payload = vec![0u8; payload_len];
            reader.read_exact(&mut payload)?;

            if let Ok(rec) = self.deserialize_record(&payload) {
                records.insert(rec.id.clone(), rec);
            }
        }

        Ok(log_position)
    }

    /// Replay log entries starting from the given position.
    fn replay_log(&self, records: &mut HashMap<String, Record>, start_pos: u64) -> Result<()> {
        let file = File::open(&self.log_path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(start_pos))?;

        while reader.stream_position()? < file_len {
            let op = match reader.read_u8() {
                Ok(op) => op,
                Err(_) => break,
            };

            let payload_len = match reader.read_u32::<LittleEndian>() {
                Ok(len) => len as usize,
                Err(_) => break,
            };

            let expected_crc = match reader.read_u32::<LittleEndian>() {
                Ok(crc) => crc,
                Err(_) => break,
            };

            let mut payload = vec![0u8; payload_len];
            if reader.read_exact(&mut payload).is_err() {
                break;
            }

            // Verify CRC32
            let actual_crc = crc32fast::hash(&payload);
            if actual_crc != expected_crc {
                tracing::warn!("CRC mismatch in cognitive log, skipping entry");
                continue;
            }

            match op {
                OP_STORE | OP_UPDATE => {
                    if let Ok(rec) = self.deserialize_record(&payload) {
                        records.insert(rec.id.clone(), rec);
                    }
                }
                OP_DELETE => {
                    if payload.len() >= 12 {
                        let id = String::from_utf8_lossy(&payload[..12])
                            .trim_matches('\0')
                            .to_string();
                        records.remove(&id);
                    }
                }
                _ => {
                    tracing::warn!("Unknown op code {} in cognitive log", op);
                }
            }
        }

        Ok(())
    }

    /// Append a STORE entry for a new record.
    pub fn append_store(&self, rec: &Record) -> Result<()> {
        let payload = self.serialize_record(rec)?;
        self.append_entry(OP_STORE, &payload)
    }

    /// Append an UPDATE entry for an existing record.
    pub fn append_update(&self, rec: &Record) -> Result<()> {
        let payload = self.serialize_record(rec)?;
        self.append_entry(OP_UPDATE, &payload)
    }

    /// Append a DELETE tombstone.
    pub fn append_delete(&self, record_id: &str) -> Result<()> {
        let mut id_bytes = [0u8; 12];
        let src = record_id.as_bytes();
        let len = src.len().min(12);
        id_bytes[..len].copy_from_slice(&src[..len]);
        self.append_entry(OP_DELETE, &id_bytes)
    }

    /// Low-level: append an entry to the log.
    fn append_entry(&self, op: u8, payload: &[u8]) -> Result<()> {
        let crc = crc32fast::hash(payload);

        let mut writer = self.writer.lock();
        let w = writer
            .as_mut()
            .ok_or_else(|| anyhow!("Cognitive store is closed"))?;
        w.write_u8(op)?;
        w.write_u32::<LittleEndian>(payload.len() as u32)?;
        w.write_u32::<LittleEndian>(crc)?;
        w.write_all(payload)?;
        w.flush()?;

        Ok(())
    }

    /// Write a snapshot of all current records.
    pub fn write_snapshot(&self, records: &HashMap<String, Record>) -> Result<()> {
        let log_pos = {
            let file = File::open(&self.log_path)?;
            file.metadata()?.len()
        };

        let mut writer = BufWriter::new(File::create(&self.snap_path)?);
        writer.write_all(SNAP_MAGIC)?;
        writer.write_u8(VERSION)?;
        writer.write_u64::<LittleEndian>(log_pos)?;
        writer.write_u32::<LittleEndian>(records.len() as u32)?;

        for rec in records.values() {
            let payload = self.serialize_record(rec)?;
            writer.write_u32::<LittleEndian>(payload.len() as u32)?;
            writer.write_all(&payload)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Compact: rewrite log with only live records + write snapshot.
    pub fn compact(&self, records: &HashMap<String, Record>) -> Result<()> {
        // Close writer
        {
            let mut writer = self.writer.lock();
            *writer = None;
        }

        // Rewrite log
        let temp_path = self.log_path.with_extension("tmp");
        {
            let mut f = File::create(&temp_path)?;
            f.write_all(MAGIC)?;
            f.write_u8(VERSION)?;

            for rec in records.values() {
                let payload = self.serialize_record(rec)?;
                let crc = crc32fast::hash(&payload);
                f.write_u8(OP_STORE)?;
                f.write_u32::<LittleEndian>(payload.len() as u32)?;
                f.write_u32::<LittleEndian>(crc)?;
                f.write_all(&payload)?;
            }

            f.flush()?;
        }

        fs::rename(&temp_path, &self.log_path)?;

        // Write snapshot at end of new log
        self.write_snapshot(records)?;

        // Reopen writer
        {
            let file = OpenOptions::new().append(true).open(&self.log_path)?;
            let mut writer = self.writer.lock();
            *writer = Some(BufWriter::new(file));
        }

        Ok(())
    }

    /// Flush pending writes.
    pub fn flush(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        if let Some(w) = writer.as_mut() {
            w.flush()?;
        }
        Ok(())
    }

    // ── Serialization ──

    /// Flush pending writes and release the append handle.
    pub fn close(&self) -> Result<()> {
        self.flush()?;
        let mut writer = self.writer.lock();
        writer.take();
        Ok(())
    }

    fn serialize_record(&self, rec: &Record) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(rec)?;
        Ok(json)
    }

    fn deserialize_record(&self, data: &[u8]) -> Result<Record> {
        let rec: Record = serde_json::from_slice(data)?;
        Ok(rec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::Level;

    #[test]
    fn test_store_and_load() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = CognitiveStore::new(dir.path())?;

        let mut rec = Record::new("Hello world".into(), Level::Working);
        rec.tags = vec!["test".into()];
        store.append_store(&rec)?;

        let records = store.load_all()?;
        assert_eq!(records.len(), 1);
        assert_eq!(records[&rec.id].content, "Hello world");
        Ok(())
    }

    #[test]
    fn test_update_and_delete() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = CognitiveStore::new(dir.path())?;

        let mut rec = Record::new("original".into(), Level::Working);
        store.append_store(&rec)?;

        rec.content = "updated".into();
        store.append_update(&rec)?;

        let records = store.load_all()?;
        assert_eq!(records[&rec.id].content, "updated");

        store.append_delete(&rec.id)?;
        let records = store.load_all()?;
        assert!(records.is_empty());
        Ok(())
    }

    #[test]
    fn test_snapshot_and_compact() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = CognitiveStore::new(dir.path())?;

        for i in 0..10 {
            let rec = Record::new(format!("record {}", i), Level::Working);
            store.append_store(&rec)?;
        }

        let records = store.load_all()?;
        assert_eq!(records.len(), 10);

        store.compact(&records)?;

        let records2 = store.load_all()?;
        assert_eq!(records2.len(), 10);
        Ok(())
    }

    #[test]
    fn test_close_releases_cognitive_store_handle() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store_path = dir.path().join("close_test");
        std::fs::create_dir_all(&store_path)?;

        let store = CognitiveStore::new(&store_path)?;
        let rec = Record::new("close".into(), Level::Working);
        store.append_store(&rec)?;
        store.close()?;

        std::fs::remove_dir_all(&store_path)?;
        assert!(!store_path.exists());
        Ok(())
    }
}
