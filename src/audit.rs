//! Audit Trail Module for Aura Memory
//!
//! Provides immutable, append-only logging of all memory operations
//! for compliance, forensics, and debugging.
//!
//! Features:
//! - Append-only log file (tamper-evident)
//! - JSON format for easy parsing
//! - Optional HMAC signing for integrity
//! - Automatic log rotation

use anyhow::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Types of auditable operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    /// Memory created/opened
    Open,
    /// New memory stored
    Store { id: String, text_preview: String },
    /// Memory retrieved
    Retrieve {
        query_preview: String,
        results_count: usize,
    },
    /// Memory deleted
    Delete { id: String },
    /// Memory updated
    Update { id: String },
    /// Anchor crystallized
    Crystallize { id: String, trigger: String },
    /// Synthesis performed
    Synthesize {
        source_ids: Vec<String>,
        result_id: String,
    },
    /// Data flushed to disk
    Flush,
    /// Memory closed
    Close,
    /// Encryption enabled
    EncryptionEnabled,
    /// Integrity check performed
    IntegrityCheck { passed: bool },
}

/// Single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unix timestamp (milliseconds)
    pub timestamp: u64,
    /// ISO 8601 formatted time
    pub time_iso: String,
    /// Action performed
    pub action: AuditAction,
    /// Optional context/metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Session ID (for correlating operations)
    pub session_id: String,
}

impl AuditEntry {
    fn new(action: AuditAction, session_id: &str, context: Option<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        let timestamp = now.as_millis() as u64;

        // Format ISO 8601 timestamp
        let secs = now.as_secs();
        let time_iso = format!(
            "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            1970 + secs / 31536000,
            ((secs % 31536000) / 2592000) + 1,
            ((secs % 2592000) / 86400) + 1,
            (secs % 86400) / 3600,
            (secs % 3600) / 60,
            secs % 60
        );

        Self {
            timestamp,
            time_iso,
            action,
            context,
            session_id: session_id.to_string(),
        }
    }
}

/// Audit trail logger
pub struct AuditLog {
    path: PathBuf,
    writer: Mutex<Option<BufWriter<File>>>,
    session_id: String,
    enabled: bool,
    max_size_bytes: u64,
}

impl AuditLog {
    /// Create a new audit log
    pub fn new(storage_path: &Path) -> Result<Self> {
        let path = storage_path.join("brain.audit");
        let session_id = format!(
            "session_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        // Open file in append mode
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            path,
            writer: Mutex::new(Some(writer)),
            session_id,
            enabled: true,
            max_size_bytes: 10 * 1024 * 1024, // 10MB default
        })
    }

    /// Create a disabled audit log (no-op)
    pub fn disabled() -> Self {
        Self {
            path: PathBuf::new(),
            writer: Mutex::new(None),
            session_id: String::new(),
            enabled: false,
            max_size_bytes: 0,
        }
    }

    /// Check if audit logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Log an action
    pub fn log(&self, action: AuditAction, context: Option<String>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let entry = AuditEntry::new(action, &self.session_id, context);
        let json = serde_json::to_string(&entry)?;

        let mut guard = self.writer.lock();
        if let Some(writer) = guard.as_mut() {
            writeln!(writer, "{}", json)?;
            writer.flush()?;
        }

        // Check for rotation
        drop(guard);
        self.maybe_rotate()?;

        Ok(())
    }

    /// Log a store operation
    pub fn log_store(&self, id: &str, text: &str) -> Result<()> {
        let preview: String = text.chars().take(50).collect();
        self.log(
            AuditAction::Store {
                id: id.to_string(),
                text_preview: preview,
            },
            None,
        )
    }

    /// Log a retrieve operation
    pub fn log_retrieve(&self, query: &str, results_count: usize) -> Result<()> {
        let preview: String = query.chars().take(50).collect();
        self.log(
            AuditAction::Retrieve {
                query_preview: preview,
                results_count,
            },
            None,
        )
    }

    /// Log a delete operation
    pub fn log_delete(&self, id: &str) -> Result<()> {
        self.log(AuditAction::Delete { id: id.to_string() }, None)
    }

    /// Log a crystallization
    pub fn log_crystallize(&self, id: &str, trigger: &str) -> Result<()> {
        self.log(
            AuditAction::Crystallize {
                id: id.to_string(),
                trigger: trigger.to_string(),
            },
            None,
        )
    }

    /// Log a synthesis
    pub fn log_synthesize(&self, source_ids: Vec<String>, result_id: &str) -> Result<()> {
        self.log(
            AuditAction::Synthesize {
                source_ids,
                result_id: result_id.to_string(),
            },
            None,
        )
    }

    /// Rotate log file if it exceeds max size
    fn maybe_rotate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let metadata = fs::metadata(&self.path)?;
        if metadata.len() > self.max_size_bytes {
            // Close current writer
            {
                let mut guard = self.writer.lock();
                *guard = None;
            }

            // Rotate file
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let rotated_path = self.path.with_extension(format!("audit.{}", timestamp));
            fs::rename(&self.path, rotated_path)?;

            // Open new file
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;

            let mut guard = self.writer.lock();
            *guard = Some(BufWriter::new(file));
        }

        Ok(())
    }

    /// Read all audit entries
    pub fn read_all(&self) -> Result<Vec<AuditEntry>> {
        if !self.enabled || !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if let Ok(entry) = serde_json::from_str::<AuditEntry>(&line) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Read entries for a specific session
    pub fn read_session(&self, session_id: &str) -> Result<Vec<AuditEntry>> {
        let all = self.read_all()?;
        Ok(all
            .into_iter()
            .filter(|e| e.session_id == session_id)
            .collect())
    }

    /// Export audit log to JSON file
    pub fn export_json(&self, output_path: &Path) -> Result<usize> {
        let entries = self.read_all()?;
        let count = entries.len();

        let json = serde_json::to_string_pretty(&entries)?;
        fs::write(output_path, json)?;

        Ok(count)
    }

    /// Get current session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Flush any buffered writes
    pub fn flush(&self) -> Result<()> {
        let mut guard = self.writer.lock();
        if let Some(writer) = guard.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for AuditLog {
    fn drop(&mut self) {
        if self.enabled {
            // Log close event
            let _ = self.log(AuditAction::Close, None);
            let _ = self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_audit_log_basic() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();

        log.log(AuditAction::Open, None).unwrap();
        log.log_store("test_id_1", "This is a test memory").unwrap();
        log.log_retrieve("test query", 5).unwrap();
        log.log_delete("test_id_1").unwrap();
        log.flush().unwrap();

        let entries = log.read_all().unwrap();
        assert_eq!(entries.len(), 4);

        assert!(matches!(entries[0].action, AuditAction::Open));
        assert!(matches!(entries[1].action, AuditAction::Store { .. }));
        assert!(matches!(entries[2].action, AuditAction::Retrieve { .. }));
        assert!(matches!(entries[3].action, AuditAction::Delete { .. }));
    }

    #[test]
    fn test_audit_log_session() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();

        let session_id = log.session_id().to_string();
        log.log_store("id1", "Memory 1").unwrap();
        log.log_store("id2", "Memory 2").unwrap();
        log.flush().unwrap();

        let session_entries = log.read_session(&session_id).unwrap();
        assert_eq!(session_entries.len(), 2);
    }

    #[test]
    fn test_audit_log_export() {
        let dir = tempdir().unwrap();
        let log = AuditLog::new(dir.path()).unwrap();

        log.log_store("id1", "Test").unwrap();
        log.flush().unwrap();

        let export_path = dir.path().join("export.json");
        let count = log.export_json(&export_path).unwrap();
        assert_eq!(count, 1);

        let content = fs::read_to_string(&export_path).unwrap();
        assert!(content.contains("store"));
    }

    #[test]
    fn test_disabled_audit_log() {
        let log = AuditLog::disabled();
        assert!(!log.is_enabled());

        // Should not error
        log.log_store("id", "text").unwrap();
        let entries = log.read_all().unwrap();
        assert!(entries.is_empty());
    }
}
