//! Deterministic structural relation helpers.
//!
//! Math-first relation logic lives here: no embeddings, no LLM, only explicit
//! symbolic signals over records.

use serde::{Deserialize, Serialize};

/// Strong deterministic weight for explicit family relations.
pub const STRUCTURAL_FAMILY_WEIGHT: f32 = 0.95;
/// Strong deterministic weight for explicit project membership relations.
pub const STRUCTURAL_PROJECT_WEIGHT: f32 = 0.92;
pub const PROJECT_MEMBERSHIP_RELATION: &str = "belongs_to_project";

const FAMILY_PATTERNS: [(&str, &str); 12] = [
    ("my brother", "family.brother"),
    ("my sister", "family.sister"),
    ("my mother", "family.mother"),
    ("my father", "family.father"),
    ("my mom", "family.mother"),
    ("my dad", "family.father"),
    ("my wife", "family.wife"),
    ("my husband", "family.husband"),
    ("my son", "family.son"),
    ("my daughter", "family.daughter"),
    ("my grandmother", "family.grandmother"),
    ("my grandfather", "family.grandfather"),
];

/// Public inspect shape for deterministic structural relations.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralRelation {
    pub source_record_id: String,
    pub target_record_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub namespace: String,
    pub evidence_record_ids: Vec<String>,
}

/// Public inspect shape for any explicit typed relation edge.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationEdge {
    pub source_record_id: String,
    pub target_record_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub namespace: String,
    pub structural: bool,
}

/// Combined inspect object for one record and its direct typed-relation corridor.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDigest {
    pub anchor_record_id: String,
    pub namespace: String,
    pub anchor_tags: Vec<String>,
    pub anchor_content: String,
    pub relation_count: usize,
    pub structural_relations: usize,
    pub non_structural_relations: usize,
    pub relation_types: std::collections::HashMap<String, usize>,
    pub linked_record_ids: Vec<String>,
    pub edges: Vec<RelationEdge>,
}

/// Deterministic aggregate for all records carrying the same local entity id.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDigest {
    pub entity_id: String,
    pub namespace: String,
    pub record_ids: Vec<String>,
    pub relation_count: usize,
    pub tags: std::collections::HashMap<String, usize>,
    pub levels: std::collections::HashMap<String, usize>,
}

/// Deterministic typed edge between two local entities.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationEdge {
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub namespace: String,
    pub source_record_id: String,
    pub target_record_id: String,
}

/// One neighbor row in a deterministic entity graph snapshot.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityGraphNeighbor {
    pub entity_id: String,
    pub anchor_record_id: String,
    pub record_count: usize,
    pub relation_count: usize,
    pub relation_types: std::collections::HashMap<String, usize>,
    pub strongest_weight: f32,
}

/// Combined inspect object for one entity and its direct entity neighbors.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityGraphDigest {
    pub entity: EntityDigest,
    pub anchor_record_id: String,
    pub neighbor_count: usize,
    pub relation_types: std::collections::HashMap<String, usize>,
    pub neighbors: Vec<EntityGraphNeighbor>,
    pub edges: Vec<EntityRelationEdge>,
}

/// One inspect row in a deterministic self/family graph.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyRelationMember {
    pub record_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub tags: Vec<String>,
    pub content: String,
}

/// Inspect-friendly snapshot of deterministic self/family relations.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyGraphSnapshot {
    pub namespace: String,
    pub profile_record_id: String,
    pub relation_count: usize,
    pub relation_types: std::collections::HashMap<String, usize>,
    pub members: Vec<FamilyRelationMember>,
}

/// Deterministic inspect object for one linked person inside the family graph.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonDigest {
    pub namespace: String,
    pub profile_record_id: String,
    pub person_record_id: String,
    pub relation_type: String,
    pub weight: f32,
    pub person_tags: Vec<String>,
    pub person_content: String,
}

/// Inspect-friendly snapshot of a project-scoped structural graph.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectGraphSnapshot {
    pub project_id: String,
    pub project_record_id: String,
    pub project_topic: String,
    pub namespace: String,
    pub member_record_ids: Vec<String>,
    pub member_tags: std::collections::HashMap<String, usize>,
    pub relation_count: usize,
}

/// Deterministic project status counters derived from project-scoped records.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStatusSnapshot {
    pub project_id: String,
    pub project_record_id: String,
    pub project_topic: String,
    pub namespace: String,
    pub project_status: String,
    pub total_members: usize,
    pub reports: usize,
    pub scheduled_tasks: usize,
    pub open_tasks: usize,
    pub completed_tasks: usize,
    pub todos: usize,
    pub open_todos: usize,
    pub completed_todos: usize,
    pub notes: usize,
    pub due_tasks: usize,
    pub high_priority_todos: usize,
}

/// One deterministic timeline row for a project-scoped record.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTimelineEntry {
    pub record_id: String,
    pub content: String,
    pub kind: String,
    pub status: String,
    pub created_at: f64,
    pub due_date: Option<String>,
    pub overdue: bool,
    pub tags: Vec<String>,
}

/// Deterministic project timeline snapshot.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTimelineSnapshot {
    pub project_id: String,
    pub project_record_id: String,
    pub project_topic: String,
    pub namespace: String,
    pub total_entries: usize,
    pub overdue_entries: usize,
    pub upcoming_entries: usize,
    pub open_entries: usize,
    pub entries: Vec<ProjectTimelineEntry>,
}

/// Combined deterministic project digest for runtime/UI inspection.
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDigest {
    pub graph: ProjectGraphSnapshot,
    pub status: ProjectStatusSnapshot,
    pub timeline: ProjectTimelineSnapshot,
}

/// Normalize free text into a lowercase alphanumeric corridor.
pub fn normalize_relation_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len() + 2);
    normalized.push(' ');
    let mut last_was_space = true;
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch);
            last_was_space = false;
        } else if !last_was_space {
            normalized.push(' ');
            last_was_space = true;
        }
    }
    if !last_was_space {
        normalized.push(' ');
    }
    normalized
}

/// Detect a deterministic family relation from record content.
pub fn detect_family_relation(content: &str) -> Option<&'static str> {
    let normalized = normalize_relation_text(content);
    FAMILY_PATTERNS
        .iter()
        .find_map(|(pattern, relation)| normalized.contains(pattern).then_some(*relation))
}

/// Family relations use the `family.*` typed structural corridor.
pub fn is_family_relation_type(relation_type: &str) -> bool {
    relation_type.starts_with("family.")
}

/// Structural relations are explicit non-semantic graph edges.
pub fn is_structural_relation_type(relation_type: &str) -> bool {
    is_family_relation_type(relation_type) || relation_type == PROJECT_MEMBERSHIP_RELATION
}
