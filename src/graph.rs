//! Knowledge graph & session tracking.
//!
//! Rewritten from aura-cognitive graph.py.
//! Manages connections between records and session-based co-activation.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::record::Record;
use crate::ngram::NGramIndex;
use crate::cognitive_store::CognitiveStore;

/// Maximum connections per record.
pub const MAX_CONNECTIONS: usize = 50;

/// Session timeout in seconds (30 minutes).
const SESSION_TIMEOUT: f64 = 1800.0;

/// Ephemeral session buffer for co-activation tracking.
#[derive(Debug, Clone)]
pub struct SessionBuffer {
    pub record_ids: HashSet<String>,
    pub started_at: f64,
    pub last_activity: f64,
}

impl Default for SessionBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionBuffer {
    pub fn new() -> Self {
        let now = now_secs();
        Self {
            record_ids: HashSet::new(),
            started_at: now,
            last_activity: now,
        }
    }
}

/// Manages session-scoped co-activation tracking.
pub struct SessionTracker {
    sessions: HashMap<String, SessionBuffer>,
}

impl Default for SessionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionTracker {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Track that these record IDs were activated in a session.
    pub fn track_activation(&mut self, session_id: &str, record_ids: &[String]) {
        let buf = self.sessions
            .entry(session_id.to_string())
            .or_default();

        buf.record_ids.extend(record_ids.iter().cloned());
        buf.last_activity = now_secs();
    }

    /// End a session and return co-activation strengthening stats.
    pub fn end_session(
        &mut self,
        session_id: &str,
        records: &mut HashMap<String, Record>,
    ) -> HashMap<String, usize> {
        let buf = match self.sessions.remove(session_id) {
            Some(b) => b,
            None => return HashMap::new(),
        };

        self.consolidate_session(&buf, records)
    }

    /// Strengthen connections between all records in a session.
    fn consolidate_session(
        &self,
        buf: &SessionBuffer,
        records: &mut HashMap<String, Record>,
    ) -> HashMap<String, usize> {
        let ids: Vec<String> = buf.record_ids.iter().cloned().collect();
        let mut pairs_strengthened = 0;

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let id_a = &ids[i];
                let id_b = &ids[j];

                // Namespace guard: skip cross-namespace co-activation
                let same_ns = match (records.get(id_a), records.get(id_b)) {
                    (Some(a), Some(b)) => a.namespace == b.namespace,
                    _ => false,
                };
                if !same_ns {
                    continue;
                }

                // Get current weight
                let current = records
                    .get(id_a)
                    .and_then(|r| r.connections.get(id_b).copied())
                    .unwrap_or(0.0);

                // Diminishing returns
                let delta = 0.05 * (1.0 - current);
                let boosted = (current + delta).min(1.0);

                // Set bidirectional with coactivation type
                if let Some(rec_a) = records.get_mut(id_a) {
                    rec_a.connections.insert(id_b.clone(), boosted);
                    rec_a.connection_types.entry(id_b.clone()).or_insert_with(|| "coactivation".to_string());
                }
                if let Some(rec_b) = records.get_mut(id_b) {
                    rec_b.connections.insert(id_a.clone(), boosted);
                    rec_b.connection_types.entry(id_a.clone()).or_insert_with(|| "coactivation".to_string());
                }

                pairs_strengthened += 1;
            }
        }

        let mut stats = HashMap::new();
        stats.insert("pairs_strengthened".to_string(), pairs_strengthened);
        stats.insert("session_records".to_string(), ids.len());
        stats
    }

    /// Remove stale sessions (inactive for > SESSION_TIMEOUT).
    pub fn cleanup_stale_sessions(&mut self, records: &mut HashMap<String, Record>) {
        let now = now_secs();
        let stale: Vec<String> = self.sessions
            .iter()
            .filter(|(_, buf)| now - buf.last_activity > SESSION_TIMEOUT)
            .map(|(id, _)| id.clone())
            .collect();

        for id in stale {
            if let Some(buf) = self.sessions.remove(&id) {
                self.consolidate_session(&buf, records);
            }
        }
    }
}

/// Auto-connect a new record with existing records that share tags.
///
/// Returns the number of connections made.
pub fn auto_connect(
    new_record: &mut Record,
    tag_index: &HashMap<String, HashSet<String>>,
    records: &mut HashMap<String, Record>,
) -> usize {
    if new_record.tags.is_empty() {
        return 0;
    }

    // Collect candidate IDs from tag index
    let mut candidates: HashMap<String, usize> = HashMap::new();
    for tag in &new_record.tags {
        if let Some(ids) = tag_index.get(tag) {
            for id in ids {
                if id != &new_record.id {
                    *candidates.entry(id.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    let mut connected = 0;
    for (candidate_id, shared_count) in &candidates {
        if new_record.connections.len() >= MAX_CONNECTIONS {
            break;
        }

        // Namespace guard: never auto-connect records across namespaces
        if let Some(candidate) = records.get(candidate_id) {
            if candidate.namespace != new_record.namespace {
                continue;
            }
        }

        let weight = (0.2 + 0.15 * *shared_count as f32).min(0.8);

        new_record.add_typed_connection(candidate_id, weight, "associative");

        if let Some(other) = records.get_mut(candidate_id) {
            if other.connections.len() < MAX_CONNECTIONS {
                other.add_typed_connection(&new_record.id, weight, "associative");
            }
        }

        connected += 1;
    }

    connected
}

/// Remove a record and clean up all indexes and bidirectional connections.
pub fn remove_record(
    rid: &str,
    records: &mut HashMap<String, Record>,
    ngram_index: &mut NGramIndex,
    tag_index: &mut HashMap<String, HashSet<String>>,
    aura_index: &mut HashMap<String, String>,
    store: &CognitiveStore,
) {
    // Remove from ngram index
    ngram_index.remove(rid);

    // Remove from tag index
    if let Some(rec) = records.get(rid) {
        for tag in &rec.tags {
            if let Some(ids) = tag_index.get_mut(tag) {
                ids.remove(rid);
            }
        }

        // Remove from aura index
        if let Some(ref aura_id) = rec.aura_id {
            aura_index.remove(aura_id);
        }

        // Remove bidirectional connections (weights + types)
        let conn_ids: Vec<String> = rec.connections.keys().cloned().collect();
        for conn_id in conn_ids {
            if let Some(other) = records.get_mut(&conn_id) {
                other.connections.remove(rid);
                other.connection_types.remove(rid);
            }
        }
    }

    // Remove from records
    records.remove(rid);

    // Persist deletion
    let _ = store.append_delete(rid);
}

/// Merge 'remove' record into 'keep' record.
pub fn merge_records(
    keep_id: &str,
    remove_id: &str,
    records: &mut HashMap<String, Record>,
    ngram_index: &mut NGramIndex,
    tag_index: &mut HashMap<String, HashSet<String>>,
    aura_index: &mut HashMap<String, String>,
    store: &CognitiveStore,
) {
    // Get data from the record being removed
    let (remove_tags, remove_connections, remove_conn_types, remove_strength, remove_level, remove_activation, remove_source_type) = {
        if let Some(remove) = records.get(remove_id) {
            (
                remove.tags.clone(),
                remove.connections.clone(),
                remove.connection_types.clone(),
                remove.strength,
                remove.level,
                remove.activation_count,
                remove.source_type.clone(),
            )
        } else {
            return;
        }
    };

    // Merge into keep
    if let Some(keep) = records.get_mut(keep_id) {
        // Upgrade level
        if remove_level > keep.level {
            keep.level = remove_level;
        }

        // Merge tags
        for tag in &remove_tags {
            if !keep.tags.contains(tag) {
                keep.tags.push(tag.clone());
            }
        }

        // Transfer connections and types (remap remove→keep pointers)
        for (conn_id, weight) in &remove_connections {
            if conn_id != keep_id && keep.connections.len() < MAX_CONNECTIONS {
                let existing = keep.connections.get(conn_id).copied().unwrap_or(0.0);
                keep.connections.insert(conn_id.clone(), existing.max(*weight));
                // Preserve relationship type from removed record if keep doesn't have one
                if let Some(rel_type) = remove_conn_types.get(conn_id) {
                    keep.connection_types.entry(conn_id.clone()).or_insert_with(|| rel_type.clone());
                }
            }
        }

        // Combine strength
        keep.strength = (keep.strength + 0.3 * remove_strength).min(1.0);
        keep.activation_count += remove_activation;

        // Preserve higher-authority source_type (recorded > retrieved > inferred > generated)
        let rank = |st: &str| -> u8 {
            match st { "recorded" => 3, "retrieved" => 2, "inferred" => 1, _ => 0 }
        };
        if rank(&remove_source_type) > rank(&keep.source_type) {
            keep.source_type = remove_source_type;
        }
    }

    // Delete the removed record
    remove_record(remove_id, records, ngram_index, tag_index, aura_index, store);

    // Persist the updated keep record
    if let Some(keep) = records.get(keep_id) {
        let _ = store.append_update(keep);
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::levels::Level;

    #[test]
    fn test_session_tracker() {
        let mut tracker = SessionTracker::new();
        let mut records = HashMap::new();

        let r1 = Record::new("test1".into(), Level::Working);
        let r2 = Record::new("test2".into(), Level::Working);
        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        records.insert(id1.clone(), r1);
        records.insert(id2.clone(), r2);

        tracker.track_activation("s1", &[id1.clone(), id2.clone()]);
        let stats = tracker.end_session("s1", &mut records);

        assert_eq!(stats["pairs_strengthened"], 1);

        // Check connections were created
        assert!(records[&id1].connections.contains_key(&id2));
        assert!(records[&id2].connections.contains_key(&id1));
    }

    #[test]
    fn test_auto_connect() {
        let mut records = HashMap::new();
        let mut tag_index: HashMap<String, HashSet<String>> = HashMap::new();

        let mut r1 = Record::new("test1".into(), Level::Working);
        r1.tags = vec!["rust".into(), "code".into()];
        let id1 = r1.id.clone();
        for tag in &r1.tags {
            tag_index.entry(tag.clone()).or_default().insert(id1.clone());
        }
        records.insert(id1.clone(), r1);

        let mut r2 = Record::new("test2".into(), Level::Working);
        r2.tags = vec!["rust".into(), "memory".into()];

        let connected = auto_connect(&mut r2, &tag_index, &mut records);
        assert_eq!(connected, 1);
        assert!(r2.connections.contains_key(&id1));
    }
}
