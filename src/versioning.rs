//! Semantic Versioning Module - Git for Thoughts
//!
//! Provides version control for memory state:
//! - Memory snapshots at any timestamp
//! - Diff between versions
//! - Branch/merge for parallel timelines
//! - Revert to previous version
//!
//! # Architecture
//! Each snapshot stores:
//! - Full state hash (xxHash3)
//! - Delta from previous snapshot
//! - Metadata (timestamp, message, parent)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Snapshot identifier (first 12 chars of hash)
pub type SnapshotId = String;

/// Full snapshot hash (64-bit xxHash3 as hex)
pub type SnapshotHash = String;

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Short ID (first 12 chars of hash)
    pub id: SnapshotId,
    /// Full hash of snapshot content
    pub hash: SnapshotHash,
    /// Parent snapshot ID (None for initial)
    pub parent: Option<SnapshotId>,
    /// Creation timestamp (Unix ms)
    pub timestamp: u64,
    /// ISO 8601 timestamp
    pub timestamp_iso: String,
    /// User-provided commit message
    pub message: String,
    /// Branch name (default: "main")
    pub branch: String,
    /// Number of records in this snapshot
    pub record_count: u64,
    /// Total size in bytes
    pub size_bytes: u64,
    /// Tags for this snapshot
    pub tags: Vec<String>,
}

/// Record change type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
}

/// Single record change in a diff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordChange {
    /// Record ID
    pub id: String,
    /// Type of change
    pub change_type: ChangeType,
    /// Old text (for Modified/Deleted)
    pub old_text: Option<String>,
    /// New text (for Added/Modified)
    pub new_text: Option<String>,
    /// Old timestamp
    pub old_timestamp: Option<u64>,
    /// New timestamp
    pub new_timestamp: Option<u64>,
}

/// Diff between two snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Source snapshot ID
    pub from_id: SnapshotId,
    /// Target snapshot ID
    pub to_id: SnapshotId,
    /// List of changes
    pub changes: Vec<RecordChange>,
    /// Number of added records
    pub added: usize,
    /// Number of modified records
    pub modified: usize,
    /// Number of deleted records
    pub deleted: usize,
}

impl VersionDiff {
    /// Check if diff is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "+{} ~{} -{} ({} changes)",
            self.added,
            self.modified,
            self.deleted,
            self.changes.len()
        )
    }
}

/// Branch information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch name
    pub name: String,
    /// Current HEAD snapshot ID
    pub head: SnapshotId,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

/// Version history entry for display
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub snapshot: Snapshot,
    pub is_head: bool,
    pub branch_name: String,
}

/// Memory record for versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedRecord {
    pub id: String,
    pub text: String,
    pub timestamp: u64,
    pub layer: String,
}

/// Version control manager
#[allow(dead_code)]
pub struct VersionManager {
    /// Storage directory for versions
    storage_path: PathBuf,
    /// All snapshots indexed by ID
    snapshots: HashMap<SnapshotId, Snapshot>,
    /// All branches
    branches: HashMap<String, Branch>,
    /// Current branch name
    current_branch: String,
    /// Dirty flag
    dirty: bool,
}

impl VersionManager {
    /// Create or load version manager
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();
        let versions_dir = storage_path.join("versions");
        fs::create_dir_all(&versions_dir)?;

        let mut manager = Self {
            storage_path,
            snapshots: HashMap::new(),
            branches: HashMap::new(),
            current_branch: "main".to_string(),
            dirty: false,
        };

        manager.load_index()?;

        // Ensure main branch exists
        if !manager.branches.contains_key("main") {
            manager.branches.insert(
                "main".to_string(),
                Branch {
                    name: "main".to_string(),
                    head: String::new(),
                    created_at: Self::now_ms(),
                    updated_at: Self::now_ms(),
                },
            );
        }

        Ok(manager)
    }

    /// Get current timestamp in milliseconds
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Format timestamp as ISO 8601
    fn format_iso(timestamp_ms: u64) -> String {
        let secs = timestamp_ms / 1000;
        format!(
            "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            1970 + secs / 31536000,
            ((secs % 31536000) / 2592000) + 1,
            ((secs % 2592000) / 86400) + 1,
            (secs % 86400) / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    }

    /// Load snapshot index from disk
    fn load_index(&mut self) -> Result<()> {
        let index_path = self.storage_path.join("versions").join("index.json");

        if index_path.exists() {
            let content = fs::read_to_string(&index_path)?;
            let index: VersionIndex = serde_json::from_str(&content)?;

            self.snapshots = index.snapshots;
            self.branches = index.branches;
            self.current_branch = index.current_branch;
        }

        Ok(())
    }

    /// Save snapshot index to disk
    fn save_index(&self) -> Result<()> {
        let index_path = self.storage_path.join("versions").join("index.json");

        let index = VersionIndex {
            snapshots: self.snapshots.clone(),
            branches: self.branches.clone(),
            current_branch: self.current_branch.clone(),
        };

        let content = serde_json::to_string_pretty(&index)?;
        fs::write(&index_path, content)?;

        Ok(())
    }

    /// Create a new snapshot from current state
    pub fn create_snapshot(
        &mut self,
        records: &[VersionedRecord],
        message: impl Into<String>,
    ) -> Result<Snapshot> {
        let message = message.into();
        let timestamp = Self::now_ms();

        // Serialize records
        let records_json = serde_json::to_vec(records)?;
        let size_bytes = records_json.len() as u64;

        // Compute hash
        let hash = format!("{:016x}", xxhash_rust::xxh3::xxh3_64(&records_json));
        let id = hash[..12].to_string();

        // Check for duplicate
        if self.snapshots.contains_key(&id) {
            return Err(anyhow!("Snapshot {} already exists (no changes?)", id));
        }

        // Get parent from current branch
        let parent = self.branches.get(&self.current_branch).and_then(|b| {
            if b.head.is_empty() {
                None
            } else {
                Some(b.head.clone())
            }
        });

        let snapshot = Snapshot {
            id: id.clone(),
            hash,
            parent,
            timestamp,
            timestamp_iso: Self::format_iso(timestamp),
            message,
            branch: self.current_branch.clone(),
            record_count: records.len() as u64,
            size_bytes,
            tags: Vec::new(),
        };

        // Save snapshot data
        let snapshot_path = self
            .storage_path
            .join("versions")
            .join(format!("{}.snap", id));

        let file = File::create(&snapshot_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&records_json)?;

        // Update index
        self.snapshots.insert(id.clone(), snapshot.clone());

        // Update branch HEAD
        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = id;
            branch.updated_at = timestamp;
        }

        self.save_index()?;

        Ok(snapshot)
    }

    /// Get snapshot by ID (supports partial ID)
    pub fn get_snapshot(&self, id: &str) -> Option<&Snapshot> {
        // Try exact match first
        if let Some(s) = self.snapshots.get(id) {
            return Some(s);
        }

        // Try prefix match
        for (key, snapshot) in &self.snapshots {
            if key.starts_with(id) {
                return Some(snapshot);
            }
        }

        None
    }

    /// Load records from a snapshot
    pub fn load_snapshot(&self, id: &str) -> Result<Vec<VersionedRecord>> {
        let snapshot = self
            .get_snapshot(id)
            .ok_or_else(|| anyhow!("Snapshot not found: {}", id))?;

        let snapshot_path = self
            .storage_path
            .join("versions")
            .join(format!("{}.snap", snapshot.id));

        let file = File::open(&snapshot_path)?;
        let reader = BufReader::new(file);
        let records: Vec<VersionedRecord> = serde_json::from_reader(reader)?;

        Ok(records)
    }

    /// Compute diff between two snapshots
    pub fn diff(&self, from_id: &str, to_id: &str) -> Result<VersionDiff> {
        let from_records = self.load_snapshot(from_id)?;
        let to_records = self.load_snapshot(to_id)?;

        // Index records by ID
        let from_map: HashMap<&str, &VersionedRecord> =
            from_records.iter().map(|r| (r.id.as_str(), r)).collect();

        let to_map: HashMap<&str, &VersionedRecord> =
            to_records.iter().map(|r| (r.id.as_str(), r)).collect();

        let mut changes = Vec::new();
        let mut added = 0;
        let mut modified = 0;
        let mut deleted = 0;

        // Find added and modified
        for (id, to_rec) in &to_map {
            match from_map.get(id) {
                Some(from_rec) => {
                    // Check if modified
                    if from_rec.text != to_rec.text {
                        changes.push(RecordChange {
                            id: id.to_string(),
                            change_type: ChangeType::Modified,
                            old_text: Some(from_rec.text.clone()),
                            new_text: Some(to_rec.text.clone()),
                            old_timestamp: Some(from_rec.timestamp),
                            new_timestamp: Some(to_rec.timestamp),
                        });
                        modified += 1;
                    }
                }
                None => {
                    // Added
                    changes.push(RecordChange {
                        id: id.to_string(),
                        change_type: ChangeType::Added,
                        old_text: None,
                        new_text: Some(to_rec.text.clone()),
                        old_timestamp: None,
                        new_timestamp: Some(to_rec.timestamp),
                    });
                    added += 1;
                }
            }
        }

        // Find deleted
        for (id, from_rec) in &from_map {
            if !to_map.contains_key(id) {
                changes.push(RecordChange {
                    id: id.to_string(),
                    change_type: ChangeType::Deleted,
                    old_text: Some(from_rec.text.clone()),
                    new_text: None,
                    old_timestamp: Some(from_rec.timestamp),
                    new_timestamp: None,
                });
                deleted += 1;
            }
        }

        // Resolve full IDs
        let from_snapshot = self
            .get_snapshot(from_id)
            .ok_or_else(|| anyhow!("Snapshot not found: {}", from_id))?;
        let to_snapshot = self
            .get_snapshot(to_id)
            .ok_or_else(|| anyhow!("Snapshot not found: {}", to_id))?;

        Ok(VersionDiff {
            from_id: from_snapshot.id.clone(),
            to_id: to_snapshot.id.clone(),
            changes,
            added,
            modified,
            deleted,
        })
    }

    /// Get history of snapshots (newest first)
    pub fn history(&self, limit: usize) -> Vec<HistoryEntry> {
        let head_id = self
            .branches
            .get(&self.current_branch)
            .map(|b| b.head.clone())
            .unwrap_or_default();

        // Walk the parent chain from HEAD for correct ordering (newest first)
        let mut entries = Vec::new();
        let mut current_id = head_id.clone();

        while !current_id.is_empty() && (limit == 0 || entries.len() < limit) {
            if let Some(snap) = self.snapshots.get(&current_id) {
                if snap.branch == self.current_branch {
                    entries.push(HistoryEntry {
                        snapshot: snap.clone(),
                        is_head: snap.id == head_id,
                        branch_name: snap.branch.clone(),
                    });
                    current_id = snap.parent.clone().unwrap_or_default();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        entries
    }

    /// Get all snapshots for a branch
    pub fn branch_history(&self, branch_name: &str) -> Vec<&Snapshot> {
        let mut snapshots: Vec<_> = self
            .snapshots
            .values()
            .filter(|s| s.branch == branch_name)
            .collect();

        snapshots.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        snapshots
    }

    /// Create a new branch from current HEAD
    pub fn create_branch(&mut self, name: &str) -> Result<Branch> {
        if self.branches.contains_key(name) {
            return Err(anyhow!("Branch '{}' already exists", name));
        }

        let current_head = self
            .branches
            .get(&self.current_branch)
            .map(|b| b.head.clone())
            .unwrap_or_default();

        let now = Self::now_ms();
        let branch = Branch {
            name: name.to_string(),
            head: current_head,
            created_at: now,
            updated_at: now,
        };

        self.branches.insert(name.to_string(), branch.clone());
        self.save_index()?;

        Ok(branch)
    }

    /// Switch to a different branch
    pub fn checkout_branch(&mut self, name: &str) -> Result<()> {
        if !self.branches.contains_key(name) {
            return Err(anyhow!("Branch '{}' not found", name));
        }

        self.current_branch = name.to_string();
        self.save_index()?;

        Ok(())
    }

    /// Delete a branch
    pub fn delete_branch(&mut self, name: &str) -> Result<()> {
        if name == "main" {
            return Err(anyhow!("Cannot delete 'main' branch"));
        }

        if name == self.current_branch {
            return Err(anyhow!("Cannot delete current branch"));
        }

        self.branches.remove(name);
        self.save_index()?;

        Ok(())
    }

    /// List all branches
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }

    /// Get current branch name
    pub fn current_branch(&self) -> &str {
        &self.current_branch
    }

    /// Get current HEAD snapshot
    pub fn head(&self) -> Option<&Snapshot> {
        let head_id = self
            .branches
            .get(&self.current_branch)
            .map(|b| b.head.as_str())?;

        if head_id.is_empty() {
            return None;
        }

        self.get_snapshot(head_id)
    }

    /// Tag a snapshot
    pub fn tag_snapshot(&mut self, id: &str, tag: &str) -> Result<()> {
        let snapshot = self
            .snapshots
            .get_mut(id)
            .ok_or_else(|| anyhow!("Snapshot not found: {}", id))?;

        if !snapshot.tags.contains(&tag.to_string()) {
            snapshot.tags.push(tag.to_string());
        }

        self.save_index()?;
        Ok(())
    }

    /// Find snapshot by tag
    pub fn find_by_tag(&self, tag: &str) -> Option<&Snapshot> {
        self.snapshots
            .values()
            .find(|s| s.tags.contains(&tag.to_string()))
    }

    /// Get snapshot count
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get total storage size
    pub fn total_size(&self) -> u64 {
        self.snapshots.values().map(|s| s.size_bytes).sum()
    }

    /// Garbage collect orphaned snapshots
    pub fn gc(&mut self) -> Result<usize> {
        // Find all reachable snapshots from branch heads
        let mut reachable = HashSet::new();

        for branch in self.branches.values() {
            let mut current = Some(branch.head.clone());
            while let Some(id) = current {
                if id.is_empty() || reachable.contains(&id) {
                    break;
                }
                reachable.insert(id.clone());
                current = self.snapshots.get(&id).and_then(|s| s.parent.clone());
            }
        }

        // Also keep tagged snapshots
        for snapshot in self.snapshots.values() {
            if !snapshot.tags.is_empty() {
                reachable.insert(snapshot.id.clone());
            }
        }

        // Remove unreachable snapshots
        let to_remove: Vec<_> = self
            .snapshots
            .keys()
            .filter(|id| !reachable.contains(*id))
            .cloned()
            .collect();

        let count = to_remove.len();
        for id in &to_remove {
            self.snapshots.remove(id);

            // Delete snapshot file
            let snapshot_path = self
                .storage_path
                .join("versions")
                .join(format!("{}.snap", id));
            let _ = fs::remove_file(snapshot_path);
        }

        if count > 0 {
            self.save_index()?;
        }

        Ok(count)
    }
}

/// Serializable version index
#[derive(Debug, Serialize, Deserialize)]
struct VersionIndex {
    snapshots: HashMap<SnapshotId, Snapshot>,
    branches: HashMap<String, Branch>,
    current_branch: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_records(texts: &[&str]) -> Vec<VersionedRecord> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| VersionedRecord {
                id: format!("rec_{}", i),
                text: text.to_string(),
                timestamp: 1000000 + i as u64,
                layer: "general".to_string(),
            })
            .collect()
    }

    #[test]
    fn test_create_snapshot() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        let records = make_records(&["Hello world", "Test memory"]);
        let snapshot = vm.create_snapshot(&records, "Initial commit").unwrap();

        assert_eq!(snapshot.record_count, 2);
        assert_eq!(snapshot.message, "Initial commit");
        assert_eq!(snapshot.branch, "main");
        assert!(snapshot.parent.is_none());
    }

    #[test]
    fn test_load_snapshot() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        let records = make_records(&["Hello world", "Test memory"]);
        let snapshot = vm.create_snapshot(&records, "Test").unwrap();

        let loaded = vm.load_snapshot(&snapshot.id).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text, "Hello world");
    }

    #[test]
    fn test_diff() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        // First snapshot
        let records1 = make_records(&["Hello world", "Test memory"]);
        let snap1 = vm.create_snapshot(&records1, "v1").unwrap();

        // Second snapshot with changes
        let records2 = vec![
            VersionedRecord {
                id: "rec_0".to_string(),
                text: "Hello world modified".to_string(), // Modified
                timestamp: 2000000,
                layer: "general".to_string(),
            },
            // rec_1 deleted
            VersionedRecord {
                id: "rec_2".to_string(),
                text: "New record".to_string(), // Added
                timestamp: 2000001,
                layer: "general".to_string(),
            },
        ];
        let snap2 = vm.create_snapshot(&records2, "v2").unwrap();

        let diff = vm.diff(&snap1.id, &snap2.id).unwrap();
        assert_eq!(diff.added, 1);
        assert_eq!(diff.modified, 1);
        assert_eq!(diff.deleted, 1);
    }

    #[test]
    fn test_history() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        vm.create_snapshot(&make_records(&["v1"]), "First").unwrap();
        vm.create_snapshot(&make_records(&["v2"]), "Second")
            .unwrap();
        vm.create_snapshot(&make_records(&["v3"]), "Third").unwrap();

        let history = vm.history(10);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].snapshot.message, "Third"); // Newest first
        assert!(history[0].is_head);
    }

    #[test]
    fn test_branches() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        // Create initial commit on main
        vm.create_snapshot(&make_records(&["main v1"]), "Main commit")
            .unwrap();

        // Create and switch to feature branch
        vm.create_branch("feature").unwrap();
        vm.checkout_branch("feature").unwrap();

        // Commit on feature branch
        let feature_snap = vm
            .create_snapshot(&make_records(&["feature v1"]), "Feature commit")
            .unwrap();
        assert_eq!(feature_snap.branch, "feature");

        // Switch back to main
        vm.checkout_branch("main").unwrap();
        assert_eq!(vm.current_branch(), "main");

        // List branches
        let branches = vm.list_branches();
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn test_tags() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        let snap = vm
            .create_snapshot(&make_records(&["release"]), "Release")
            .unwrap();
        vm.tag_snapshot(&snap.id, "v1.0.0").unwrap();

        let found = vm.find_by_tag("v1.0.0").unwrap();
        assert_eq!(found.id, snap.id);
    }

    #[test]
    fn test_gc() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        // Create some snapshots
        vm.create_snapshot(&make_records(&["v1"]), "v1").unwrap();
        vm.create_snapshot(&make_records(&["v2"]), "v2").unwrap();
        let _snap3 = vm.create_snapshot(&make_records(&["v3"]), "v3").unwrap();

        // All should be reachable
        let removed = vm.gc().unwrap();
        assert_eq!(removed, 0);

        // HEAD points to snap3, so all are reachable via parent chain
        assert_eq!(vm.snapshot_count(), 3);
    }

    #[test]
    fn test_partial_id() {
        let dir = tempdir().unwrap();
        let mut vm = VersionManager::new(dir.path()).unwrap();

        let snap = vm
            .create_snapshot(&make_records(&["test"]), "Test")
            .unwrap();

        // Should find by partial ID (first 6 chars)
        let partial = &snap.id[..6];
        let found = vm.get_snapshot(partial);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, snap.id);
    }
}
