//! Living Memory — autonomous 8-phase maintenance engine.
//!
//! Rewritten from background_brain.py (generic phases only).
//! Agent-specific features (Telegram, file cleanup, knowledge sync) are NOT included.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use parking_lot::RwLock;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::levels::Level;
use crate::record::Record;
use crate::trust::TagTaxonomy;

// ── Archival Rule ──

/// Configurable archival rule for a tag category.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchivalRule {
    /// Tag to match.
    pub tag: String,
    /// Maximum age in days before deletion.
    pub max_age_days: u32,
    /// Keep at least this many most-recent records.
    pub keep_recent: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl ArchivalRule {
    #[new]
    fn py_new(tag: String, max_age_days: u32, keep_recent: usize) -> Self {
        Self { tag, max_age_days, keep_recent }
    }
}

/// Completed archival rule — only deletes completed/done items.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct CompletedArchivalRule {
    pub tag: String,
    pub max_age_days: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl CompletedArchivalRule {
    #[new]
    fn py_new(tag: String, max_age_days: u32) -> Self {
        Self { tag, max_age_days }
    }
}

// ── Maintenance Config ──

/// User-configurable maintenance settings.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    pub decay_enabled: bool,
    pub reflect_enabled: bool,
    pub insights_enabled: bool,
    pub consolidation_enabled: bool,
    pub synthesis_enabled: bool,
    pub archival_enabled: bool,
    /// Run level fix every Nth cycle.
    pub level_fix_interval: u64,
    /// Max clusters per consolidation run.
    pub max_clusters_per_run: usize,
    /// Configurable archival rules.
    pub archival_rules: Vec<ArchivalRule>,
    /// Completed-item archival rules.
    pub completed_archival_rules: Vec<CompletedArchivalRule>,
    /// Tag used for scheduled tasks (default: "scheduled-task").
    pub task_tag: String,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            decay_enabled: true,
            reflect_enabled: true,
            insights_enabled: true,
            consolidation_enabled: true,
            synthesis_enabled: true,
            archival_enabled: true,
            level_fix_interval: 10,
            max_clusters_per_run: 3,
            archival_rules: vec![
                ArchivalRule { tag: "web-search-cache".into(), max_age_days: 1, keep_recent: 0 },
                ArchivalRule { tag: "autonomous-outcome".into(), max_age_days: 7, keep_recent: 50 },
                ArchivalRule { tag: "session-summary".into(), max_age_days: 14, keep_recent: 20 },
                ArchivalRule { tag: "proactive-session".into(), max_age_days: 7, keep_recent: 20 },
                ArchivalRule { tag: "action-plan".into(), max_age_days: 14, keep_recent: 10 },
                ArchivalRule { tag: "session-reflection".into(), max_age_days: 30, keep_recent: 50 },
                ArchivalRule { tag: "research-finding".into(), max_age_days: 30, keep_recent: 100 },
                ArchivalRule { tag: "consolidated-meta".into(), max_age_days: 90, keep_recent: 200 },
                ArchivalRule { tag: "research-project".into(), max_age_days: 90, keep_recent: 50 },
                ArchivalRule { tag: "feedback-signal".into(), max_age_days: 14, keep_recent: 50 },
            ],
            completed_archival_rules: vec![
                CompletedArchivalRule { tag: "todo-item".into(), max_age_days: 30 },
                CompletedArchivalRule { tag: "scheduled-task".into(), max_age_days: 30 },
            ],
            task_tag: "scheduled-task".into(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MaintenanceConfig {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

// ── Maintenance Report ──

/// Decay phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct DecayReport {
    pub decayed: usize,
    pub archived: usize,
}

/// Reflect phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ReflectReport {
    pub promoted: usize,
    pub archived: usize,
}

/// Consolidation phase report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    pub native_merged: usize,
    pub clusters_found: usize,
    pub meta_created: usize,
}

/// Full maintenance cycle report.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone)]
pub struct MaintenanceReport {
    pub timestamp: String,
    pub decay: DecayReport,
    pub reflect: ReflectReport,
    pub insights_found: usize,
    pub consolidation: ConsolidationReport,
    pub cross_connections: usize,
    pub task_reminders: Vec<String>,
    pub records_archived: usize,
    pub total_records: usize,
}

// ── Phase Implementations ──

/// Phase 0: Level fix — downgrade records incorrectly at IDENTITY.
pub fn fix_memory_levels(
    records: &mut HashMap<String, Record>,
    taxonomy: &TagTaxonomy,
) -> HashMap<String, usize> {
    let mut stats = HashMap::new();
    stats.insert("total_identity".into(), 0);
    stats.insert("downgraded".into(), 0);
    stats.insert("kept".into(), 0);

    let identity_ids: Vec<String> = records.values()
        .filter(|r| r.level == Level::Identity)
        .map(|r| r.id.clone())
        .collect();

    *stats.get_mut("total_identity").unwrap() = identity_ids.len();

    for id in identity_ids {
        let rec = match records.get_mut(&id) {
            Some(r) => r,
            None => continue,
        };

        let rec_tags: HashSet<&str> = rec.tags.iter().map(|s| s.as_str()).collect();

        // Keep if has identity-specific tags
        if taxonomy.identity_tags.iter().any(|t| rec_tags.contains(t.as_str())) {
            *stats.get_mut("kept").unwrap() += 1;
            continue;
        }

        // Downgrade if has non-identity tags
        if taxonomy.non_identity_tags.iter().any(|t| rec_tags.contains(t.as_str())) {
            rec.level = Level::Domain;
            *stats.get_mut("downgraded").unwrap() += 1;
            continue;
        }

        // High activation = earned IDENTITY, keep
        if rec.activation_count >= 20 && rec.strength >= 0.9 {
            *stats.get_mut("kept").unwrap() += 1;
            continue;
        }

        // Default: downgrade
        rec.level = Level::Domain;
        *stats.get_mut("downgraded").unwrap() += 1;
    }

    stats
}

/// Phase 2: Guarded reflect — prevents overpromotion.
///
/// Saves original levels, runs reflect, restores violations:
/// 1. WORKING records are NEVER promoted
/// 2. Records can't reach IDENTITY without 20+ activations AND 0.9+ strength
pub fn guarded_reflect(
    records: &mut HashMap<String, Record>,
    _taxonomy: &TagTaxonomy,
) -> ReflectReport {
    // Save original levels for non-IDENTITY records
    let original_levels: HashMap<String, Level> = records.iter()
        .filter(|(_, r)| r.level != Level::Identity)
        .map(|(id, r)| (id.clone(), r.level))
        .collect();

    // Run reflect logic: promote candidates
    let mut promoted: usize = 0;
    let promotable: Vec<String> = records.values()
        .filter(|r| r.can_promote())
        .map(|r| r.id.clone())
        .collect();

    for id in &promotable {
        if let Some(rec) = records.get_mut(id) {
            if rec.promote() {
                promoted += 1;
            }
        }
    }

    // Archive dead
    let dead: Vec<String> = records.values()
        .filter(|r| !r.is_alive())
        .map(|r| r.id.clone())
        .collect();
    let archived = dead.len();
    for id in &dead {
        records.remove(id);
    }

    // Restore overpromotions
    let mut restored: usize = 0;
    for (rec_id, orig_level) in &original_levels {
        let rec = match records.get_mut(rec_id) {
            Some(r) => r,
            None => continue,
        };

        let should_restore =
            // Rule 1: WORKING must stay WORKING
            (*orig_level == Level::Working && rec.level != Level::Working)
            // Rule 2: No IDENTITY without threshold
            || (rec.level == Level::Identity
                && (rec.activation_count < 20 || rec.strength < 0.9));

        if should_restore {
            rec.level = *orig_level;
            restored += 1;
        }
    }

    if restored > 0 {
        tracing::info!(restored, "Overpromotion guard: restored records");
    }

    ReflectReport { promoted: promoted.saturating_sub(restored), archived }
}

/// Phase 5: Knowledge synthesis — 2-hop graph walk for cross-connections.
pub fn discover_cross_connections(
    records: &HashMap<String, Record>,
    max_discoveries: usize,
) -> Vec<String> {
    let mut discoveries = Vec::new();

    // Sample connected records
    let sample: Vec<&Record> = records.values()
        .filter(|r| !r.connections.is_empty() && r.is_alive())
        .take(10)
        .collect();

    for rec in sample {
        // 1-hop neighbors
        for neighbor_id in rec.connections.keys() {
            if let Some(neighbor) = records.get(neighbor_id) {
                // 2-hop: neighbor's connections
                for hop2_id in neighbor.connections.keys() {
                    if hop2_id != &rec.id
                        && !rec.connections.contains_key(hop2_id)
                        && discoveries.len() < max_discoveries
                    {
                        if let Some(hop2) = records.get(hop2_id) {
                            discoveries.push(format!(
                                "{} ← {} → {} (indirect connection)",
                                &rec.content[..rec.content.len().min(50)],
                                &neighbor.content[..neighbor.content.len().min(30)],
                                &hop2.content[..hop2.content.len().min(50)],
                            ));
                        }
                    }
                }
            }
        }

        if discoveries.len() >= max_discoveries {
            break;
        }
    }

    discoveries
}

/// Phase 6: Scheduled task check — find tasks due today or tomorrow.
pub fn check_scheduled_tasks(
    records: &HashMap<String, Record>,
    task_tag: &str,
) -> Vec<String> {
    let now = chrono::Utc::now();
    let tomorrow = now + chrono::Duration::days(1);
    let mut reminders = Vec::new();

    for rec in records.values() {
        if !rec.tags.contains(&task_tag.to_string()) {
            continue;
        }

        let status = rec.metadata.get("status").map(|s| s.as_str()).unwrap_or("");
        if status != "active" {
            continue;
        }

        let due_str = match rec.metadata.get("due_date") {
            Some(s) => s,
            None => continue,
        };

        let due_date = match chrono::DateTime::parse_from_rfc3339(due_str) {
            Ok(dt) => dt,
            Err(_) => {
                // Try ISO date without timezone
                match chrono::NaiveDate::parse_from_str(due_str, "%Y-%m-%d") {
                    Ok(d) => d.and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                        .fixed_offset(),
                    Err(_) => continue,
                }
            }
        };

        let due_naive = due_date.date_naive();
        let now_naive = now.date_naive();
        let tomorrow_naive = tomorrow.date_naive();

        if due_naive > tomorrow_naive {
            continue;
        }

        let description = rec.metadata.get("description")
            .unwrap_or(&rec.content);

        if due_naive == now_naive {
            reminders.push(format!("Due today: {}", description));
        } else if due_naive == tomorrow_naive {
            reminders.push(format!("Due tomorrow: {}", description));
        } else if due_naive < now_naive {
            reminders.push(format!("Overdue: {} (was due {})", description, due_naive));
        }
    }

    reminders
}

/// Phase 7: Archive old transient records.
pub fn archive_old_records(
    records: &mut HashMap<String, Record>,
    config: &MaintenanceConfig,
    taxonomy: &TagTaxonomy,
) -> usize {
    let now = chrono::Utc::now();
    let mut total_archived = 0;

    // Strategy 1: Age-based deletion
    for rule in &config.archival_rules {
        let mut matching: Vec<(String, String)> = records.values()
            .filter(|r| r.tags.contains(&rule.tag) && r.is_alive())
            .map(|r| {
                let ts = r.metadata.get("timestamp")
                    .or_else(|| r.metadata.get("created_at"))
                    .cloned()
                    .unwrap_or_default();
                (r.id.clone(), ts)
            })
            .collect();

        if matching.len() <= rule.keep_recent {
            continue;
        }

        // Sort by timestamp descending (newest first)
        matching.sort_by(|a, b| b.1.cmp(&a.1));

        let cutoff = (now - chrono::Duration::days(rule.max_age_days as i64))
            .to_rfc3339();

        // Skip keep_recent newest records
        let candidates = &matching[rule.keep_recent..];
        for (id, ts) in candidates {
            if ts.is_empty() || ts.as_str() < cutoff.as_str() {
                // Check archive protection
                if let Some(rec) = records.get(id) {
                    if !crate::guards::is_archive_protected(&rec.tags, taxonomy) {
                        records.remove(id);
                        total_archived += 1;
                    }
                }
            }
        }
    }

    // Strategy 2: Completion-based deletion
    for rule in &config.completed_archival_rules {
        let cutoff = (now - chrono::Duration::days(rule.max_age_days as i64))
            .to_rfc3339();

        let to_delete: Vec<String> = records.values()
            .filter(|r| {
                r.tags.contains(&rule.tag)
                    && matches!(
                        r.metadata.get("status").map(|s| s.as_str()),
                        Some("completed" | "done" | "cancelled" | "archived")
                    )
            })
            .filter(|r| {
                let completed_at = r.metadata.get("completed_at")
                    .or_else(|| r.metadata.get("timestamp"))
                    .or_else(|| r.metadata.get("created_at"))
                    .map(|s| s.as_str())
                    .unwrap_or("");
                completed_at.is_empty() || completed_at < cutoff.as_str()
            })
            .map(|r| r.id.clone())
            .collect();

        for id in &to_delete {
            records.remove(id);
            total_archived += 1;
        }
    }

    total_archived
}

// ── Background Brain Controller ──

/// BackgroundBrain — spawns daemon thread for periodic maintenance.
pub struct BackgroundBrain {
    /// Background loop running flag.
    running: Arc<AtomicBool>,
    /// Background thread handle.
    thread: Option<JoinHandle<()>>,
    /// Cycle count (for periodic tasks).
    pub cycle_count: AtomicU64,
    /// Transient insights from last cycle.
    pub last_insights: RwLock<Vec<String>>,
    /// Transient cross-connections from last cycle.
    pub last_cross_connections: RwLock<Vec<String>>,
}

impl BackgroundBrain {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            thread: None,
            cycle_count: AtomicU64::new(0),
            last_insights: RwLock::new(Vec::new()),
            last_cross_connections: RwLock::new(Vec::new()),
        }
    }

    /// Is the background loop currently running?
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the background loop.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    /// Get current cycle count.
    pub fn cycles(&self) -> u64 {
        self.cycle_count.load(Ordering::Relaxed)
    }
}

impl Default for BackgroundBrain {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BackgroundBrain {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_memory_levels() {
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        // Record with non-identity tag at IDENTITY → should be downgraded
        let mut r1 = Record::new("session stuff".into(), Level::Identity);
        r1.tags.push("session-summary".into());
        records.insert(r1.id.clone(), r1);

        // Record with identity tag at IDENTITY → should stay
        let mut r2 = Record::new("user profile".into(), Level::Identity);
        r2.tags.push("user-profile".into());
        records.insert(r2.id.clone(), r2);

        let stats = fix_memory_levels(&mut records, &taxonomy);
        assert_eq!(*stats.get("downgraded").unwrap(), 1);
        assert_eq!(*stats.get("kept").unwrap(), 1);
    }

    #[test]
    fn test_guarded_reflect_prevents_working_promotion() {
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        let mut r = Record::new("temporary thought".into(), Level::Working);
        r.activation_count = 10;
        r.strength = 0.9;
        let rid = r.id.clone();
        records.insert(rid.clone(), r);

        let _report = guarded_reflect(&mut records, &taxonomy);

        // WORKING should NOT have been promoted
        if let Some(rec) = records.get(&rid) {
            assert_eq!(rec.level, Level::Working);
        }
    }

    #[test]
    fn test_archival_rules() {
        let config = MaintenanceConfig::default();
        let taxonomy = TagTaxonomy::default();
        let mut records = HashMap::new();

        // Create an old cache record
        let mut r = Record::new("cached result".into(), Level::Working);
        r.tags.push("web-search-cache".into());
        r.metadata.insert("timestamp".into(), "2020-01-01T00:00:00Z".into());
        records.insert(r.id.clone(), r);

        let archived = archive_old_records(&mut records, &config, &taxonomy);
        assert_eq!(archived, 1);
        assert!(records.is_empty());
    }

    #[test]
    fn test_default_maintenance_config() {
        let config = MaintenanceConfig::default();
        assert!(config.decay_enabled);
        assert!(config.reflect_enabled);
        assert_eq!(config.level_fix_interval, 10);
        assert!(!config.archival_rules.is_empty());
    }
}
