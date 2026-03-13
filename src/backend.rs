//! Storage backend abstraction for platform-portable I/O.
//!
//! This module defines the `StorageBackend` trait that abstracts all filesystem
//! operations behind a unified interface. Two implementations are provided:
//!
//! - `FsBackend` — the default, uses `std::fs` for native builds
//! - `MemoryBackend` — in-memory storage for WASM/testing environments
//!
//! # Architecture
//!
//! All modules that need file I/O should receive an `Arc<dyn StorageBackend>`
//! instead of using `std::fs` directly. This enables:
//!
//! 1. **WASM builds** — where `std::fs` is unavailable
//! 2. **Testing** — fast in-memory backends without disk I/O
//! 3. **Custom backends** — e.g. S3, IndexedDB, or SQLite-based storage

use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Metadata about a stored file.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// Size in bytes.
    pub size: u64,
    /// Whether the file exists.
    pub exists: bool,
}

/// Abstract storage backend for all file I/O operations.
///
/// Implementors must be thread-safe (`Send + Sync`).
/// All paths are treated as opaque strings — the backend decides
/// how to interpret them (filesystem paths, keys, etc.).
pub trait StorageBackend: Send + Sync {
    // ── Read ──

    /// Read entire file as bytes.
    fn read(&self, path: &str) -> Result<Vec<u8>>;

    /// Read entire file as UTF-8 string.
    fn read_to_string(&self, path: &str) -> Result<String> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|e| anyhow::anyhow!("UTF-8 error: {}", e))
    }

    // ── Write ──

    /// Write bytes to a file, creating it if needed, truncating if exists.
    fn write(&self, path: &str, data: &[u8]) -> Result<()>;

    /// Write a UTF-8 string to a file.
    fn write_string(&self, path: &str, content: &str) -> Result<()> {
        self.write(path, content.as_bytes())
    }

    /// Append bytes to a file, creating it if needed.
    fn append(&self, path: &str, data: &[u8]) -> Result<()>;

    // ── File operations ──

    /// Check if a file or directory exists.
    fn exists(&self, path: &str) -> bool;

    /// Delete a file.
    fn remove(&self, path: &str) -> Result<()>;

    /// Rename/move a file atomically.
    fn rename(&self, from: &str, to: &str) -> Result<()>;

    /// Get file metadata (size, exists).
    fn metadata(&self, path: &str) -> Result<FileMetadata>;

    // ── Directory operations ──

    /// Create a directory and all parent directories.
    fn create_dir_all(&self, path: &str) -> Result<()>;

    /// List entries in a directory. Returns names (not full paths).
    fn list_dir(&self, path: &str) -> Result<Vec<String>>;

    // ── Durability ──

    /// Ensure all pending writes are flushed to persistent storage.
    /// On filesystem: calls fsync. On memory: no-op.
    fn sync(&self, path: &str) -> Result<()>;
}

// ═══════════════════════════════════════════════════════════
// FsBackend — default implementation using std::fs
// ═══════════════════════════════════════════════════════════

/// Filesystem-backed storage using `std::fs`.
///
/// This is the default backend for native (non-WASM) builds.
/// All operations delegate directly to the standard library.
#[derive(Debug, Clone, Default)]
pub struct FsBackend;

impl FsBackend {
    pub fn new() -> Self {
        FsBackend
    }
}

impl StorageBackend for FsBackend {
    fn read(&self, path: &str) -> Result<Vec<u8>> {
        std::fs::read(path).map_err(|e| anyhow::anyhow!("read '{}': {}", path, e))
    }

    fn read_to_string(&self, path: &str) -> Result<String> {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read_to_string '{}': {}", path, e))
    }

    fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        std::fs::write(path, data).map_err(|e| anyhow::anyhow!("write '{}': {}", path, e))
    }

    fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("append open '{}': {}", path, e))?;
        file.write_all(data)
            .map_err(|e| anyhow::anyhow!("append write '{}': {}", path, e))
    }

    fn exists(&self, path: &str) -> bool {
        std::path::Path::new(path).exists()
    }

    fn remove(&self, path: &str) -> Result<()> {
        std::fs::remove_file(path).map_err(|e| anyhow::anyhow!("remove '{}': {}", path, e))
    }

    fn rename(&self, from: &str, to: &str) -> Result<()> {
        std::fs::rename(from, to)
            .map_err(|e| anyhow::anyhow!("rename '{}' -> '{}': {}", from, to, e))
    }

    fn metadata(&self, path: &str) -> Result<FileMetadata> {
        match std::fs::metadata(path) {
            Ok(m) => Ok(FileMetadata {
                size: m.len(),
                exists: true,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(FileMetadata {
                size: 0,
                exists: false,
            }),
            Err(e) => Err(anyhow::anyhow!("metadata '{}': {}", path, e)),
        }
    }

    fn create_dir_all(&self, path: &str) -> Result<()> {
        std::fs::create_dir_all(path)
            .map_err(|e| anyhow::anyhow!("create_dir_all '{}': {}", path, e))
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)
            .map_err(|e| anyhow::anyhow!("list_dir '{}': {}", path, e))?
            .flatten()
        {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
        Ok(entries)
    }

    fn sync(&self, path: &str) -> Result<()> {
        // Open the file just to call sync_all
        if let Ok(file) = std::fs::File::open(path) {
            file.sync_all()
                .map_err(|e| anyhow::anyhow!("sync '{}': {}", path, e))?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════
// MemoryBackend — in-memory storage for WASM/testing
// ═══════════════════════════════════════════════════════════

/// In-memory storage backend.
///
/// Stores all data in a `HashMap<String, Vec<u8>>`. Useful for:
/// - WASM builds where `std::fs` is unavailable
/// - Unit tests that need fast, isolated storage
/// - Embedded environments without a filesystem
#[derive(Debug, Clone)]
pub struct MemoryBackend {
    files: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    dirs: Arc<RwLock<std::collections::HashSet<String>>>,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
            dirs: Arc::new(RwLock::new(std::collections::HashSet::new())),
        }
    }

    /// Get the number of files stored.
    pub fn file_count(&self) -> usize {
        self.files.read().len()
    }

    /// Get total bytes stored across all files.
    pub fn total_bytes(&self) -> usize {
        self.files.read().values().map(|v| v.len()).sum()
    }
}

impl StorageBackend for MemoryBackend {
    fn read(&self, path: &str) -> Result<Vec<u8>> {
        self.files
            .read()
            .get(path)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("file not found: {}", path))
    }

    fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        self.files.write().insert(path.to_string(), data.to_vec());
        Ok(())
    }

    fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        let mut files = self.files.write();
        let entry = files.entry(path.to_string()).or_default();
        entry.extend_from_slice(data);
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        self.files.read().contains_key(path) || self.dirs.read().contains(path)
    }

    fn remove(&self, path: &str) -> Result<()> {
        self.files.write().remove(path);
        Ok(())
    }

    fn rename(&self, from: &str, to: &str) -> Result<()> {
        let mut files = self.files.write();
        if let Some(data) = files.remove(from) {
            files.insert(to.to_string(), data);
            Ok(())
        } else {
            Err(anyhow::anyhow!("rename: source not found: {}", from))
        }
    }

    fn metadata(&self, path: &str) -> Result<FileMetadata> {
        let files = self.files.read();
        match files.get(path) {
            Some(data) => Ok(FileMetadata {
                size: data.len() as u64,
                exists: true,
            }),
            None => Ok(FileMetadata {
                size: 0,
                exists: false,
            }),
        }
    }

    fn create_dir_all(&self, path: &str) -> Result<()> {
        self.dirs.write().insert(path.to_string());
        Ok(())
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let prefix = if path.ends_with('/') || path.ends_with('\\') {
            path.to_string()
        } else {
            format!("{}/", path)
        };

        let files = self.files.read();
        let mut entries = Vec::new();
        for key in files.keys() {
            if let Some(rest) = key.strip_prefix(&prefix) {
                // Only direct children (no further separators)
                if !rest.contains('/') && !rest.contains('\\') {
                    entries.push(rest.to_string());
                }
            }
        }
        Ok(entries)
    }

    fn sync(&self, _path: &str) -> Result<()> {
        // No-op for in-memory storage
        Ok(())
    }
}

/// Create the default storage backend for the current platform.
pub fn default_backend() -> Arc<dyn StorageBackend> {
    Arc::new(FsBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fs_backend_basic() {
        let backend = FsBackend::new();
        let tmp = std::env::temp_dir().join("aura_backend_test");
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("test.txt");
        let path_str = path.to_str().unwrap();

        // Write
        backend.write(path_str, b"hello world").unwrap();
        assert!(backend.exists(path_str));

        // Read
        let data = backend.read(path_str).unwrap();
        assert_eq!(data, b"hello world");

        // Read string
        let text = backend.read_to_string(path_str).unwrap();
        assert_eq!(text, "hello world");

        // Metadata
        let meta = backend.metadata(path_str).unwrap();
        assert!(meta.exists);
        assert_eq!(meta.size, 11);

        // Append
        backend.append(path_str, b"!").unwrap();
        let data = backend.read(path_str).unwrap();
        assert_eq!(data, b"hello world!");

        // Rename
        let path2 = tmp.join("test2.txt");
        let path2_str = path2.to_str().unwrap();
        backend.rename(path_str, path2_str).unwrap();
        assert!(!backend.exists(path_str));
        assert!(backend.exists(path2_str));

        // Remove
        backend.remove(path2_str).unwrap();
        assert!(!backend.exists(path2_str));

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_memory_backend_basic() {
        let backend = MemoryBackend::new();

        // Write
        backend.write("test.txt", b"hello").unwrap();
        assert!(backend.exists("test.txt"));
        assert_eq!(backend.file_count(), 1);

        // Read
        let data = backend.read("test.txt").unwrap();
        assert_eq!(data, b"hello");

        // Append
        backend.append("test.txt", b" world").unwrap();
        let data = backend.read("test.txt").unwrap();
        assert_eq!(data, b"hello world");

        // Metadata
        let meta = backend.metadata("test.txt").unwrap();
        assert!(meta.exists);
        assert_eq!(meta.size, 11);

        // Non-existent file
        let meta = backend.metadata("nope.txt").unwrap();
        assert!(!meta.exists);

        // Rename
        backend.rename("test.txt", "renamed.txt").unwrap();
        assert!(!backend.exists("test.txt"));
        assert!(backend.exists("renamed.txt"));

        // Remove
        backend.remove("renamed.txt").unwrap();
        assert_eq!(backend.file_count(), 0);
    }

    #[test]
    fn test_memory_backend_directories() {
        let backend = MemoryBackend::new();

        backend.create_dir_all("data/sub").unwrap();
        assert!(backend.exists("data/sub"));

        backend.write("data/sub/a.txt", b"a").unwrap();
        backend.write("data/sub/b.txt", b"b").unwrap();
        backend.write("data/other/c.txt", b"c").unwrap();

        let entries = backend.list_dir("data/sub").unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&"a.txt".to_string()));
        assert!(entries.contains(&"b.txt".to_string()));
    }

    #[test]
    fn test_memory_backend_read_nonexistent() {
        let backend = MemoryBackend::new();
        assert!(backend.read("nope.txt").is_err());
    }

    #[test]
    fn test_memory_backend_thread_safe() {
        let backend = Arc::new(MemoryBackend::new());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let b = backend.clone();
                std::thread::spawn(move || {
                    let path = format!("file_{}.txt", i);
                    let data = format!("data_{}", i);
                    b.write(&path, data.as_bytes()).unwrap();
                    let result = b.read(&path).unwrap();
                    assert_eq!(result, data.as_bytes());
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(backend.file_count(), 10);
    }

    #[test]
    fn test_default_backend() {
        let backend = default_backend();
        // Should be FsBackend by default
        let meta = backend.metadata("/nonexistent/path/hopefully").unwrap();
        assert!(!meta.exists);
    }

    #[test]
    fn test_memory_backend_total_bytes() {
        let backend = MemoryBackend::new();
        backend.write("a.txt", b"hello").unwrap();
        backend.write("b.txt", b"world!").unwrap();
        assert_eq!(backend.total_bytes(), 11);
    }
}
