//! Concept Hierarchy (IS-A Relations)
//!
//! Implements hierarchical bit inheritance for concept graphs.
//!
//! # Math
//! If A is-a B, then SDR(A) ⊇ SDR(B) (A inherits B's semantic bits)
//!
//! # Example
//! - "dog" is-a "animal" is-a "living_thing"
//! - Query "animal" matches memories about "dog", "cat", etc.

use std::collections::{HashMap, HashSet};

/// Concept node in the hierarchy
#[derive(Clone, Debug)]
struct ConceptNode {
    /// Parent concepts (what this IS-A)
    parents: Vec<String>,
    /// Child concepts (what IS-A this)
    children: Vec<String>,
}

/// Hierarchical concept graph for IS-A relations
pub struct ConceptGraph {
    /// concept -> node
    nodes: HashMap<String, ConceptNode>,
}

impl ConceptGraph {
    /// Create a new concept graph with default ontology
    pub fn new() -> Self {
        let mut graph = Self {
            nodes: HashMap::new(),
        };
        graph.load_default_ontology();
        graph
    }

    /// Create empty graph (for custom concepts only)
    pub fn empty() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Load default concept hierarchy
    fn load_default_ontology(&mut self) {
        // Living things hierarchy
        self.add_chain(&["dog", "mammal", "animal", "living_thing", "entity"]);
        self.add_chain(&["cat", "mammal", "animal", "living_thing", "entity"]);
        self.add_chain(&["bird", "animal", "living_thing", "entity"]);
        self.add_chain(&["fish", "animal", "living_thing", "entity"]);
        self.add_chain(&["tree", "plant", "living_thing", "entity"]);
        self.add_chain(&["flower", "plant", "living_thing", "entity"]);
        self.add_chain(&["human", "mammal", "animal", "living_thing", "entity"]);
        self.add_chain(&["person", "human", "mammal", "animal", "living_thing"]);

        // Vehicles hierarchy
        self.add_chain(&["car", "automobile", "vehicle", "machine", "object"]);
        self.add_chain(&["truck", "automobile", "vehicle", "machine", "object"]);
        self.add_chain(&["motorcycle", "vehicle", "machine", "object"]);
        self.add_chain(&["bicycle", "vehicle", "object"]);
        self.add_chain(&["airplane", "aircraft", "vehicle", "machine", "object"]);
        self.add_chain(&["helicopter", "aircraft", "vehicle", "machine", "object"]);
        self.add_chain(&["boat", "watercraft", "vehicle", "object"]);
        self.add_chain(&["ship", "watercraft", "vehicle", "object"]);
        self.add_chain(&["train", "vehicle", "machine", "object"]);

        // Buildings/Places
        self.add_chain(&["house", "building", "structure", "place"]);
        self.add_chain(&["apartment", "building", "structure", "place"]);
        self.add_chain(&["office", "building", "structure", "place"]);
        self.add_chain(&["store", "building", "structure", "place"]);
        self.add_chain(&["restaurant", "building", "structure", "place"]);
        self.add_chain(&["hospital", "building", "structure", "place"]);
        self.add_chain(&["school", "building", "structure", "place"]);
        self.add_chain(&["university", "school", "building", "structure", "place"]);

        // Food hierarchy
        self.add_chain(&["apple", "fruit", "food", "consumable"]);
        self.add_chain(&["banana", "fruit", "food", "consumable"]);
        self.add_chain(&["orange", "fruit", "food", "consumable"]);
        self.add_chain(&["carrot", "vegetable", "food", "consumable"]);
        self.add_chain(&["potato", "vegetable", "food", "consumable"]);
        self.add_chain(&["bread", "baked_good", "food", "consumable"]);
        self.add_chain(&["pizza", "food", "consumable"]);
        self.add_chain(&["water", "drink", "beverage", "consumable"]);
        self.add_chain(&["coffee", "drink", "beverage", "consumable"]);
        self.add_chain(&["tea", "drink", "beverage", "consumable"]);

        // Technology
        self.add_chain(&["laptop", "computer", "device", "technology", "object"]);
        self.add_chain(&["desktop", "computer", "device", "technology", "object"]);
        self.add_chain(&["smartphone", "phone", "device", "technology", "object"]);
        self.add_chain(&["tablet", "device", "technology", "object"]);
        self.add_chain(&["keyboard", "peripheral", "device", "object"]);
        self.add_chain(&["mouse", "peripheral", "device", "object"]);
        self.add_chain(&["monitor", "display", "device", "object"]);
        self.add_chain(&["printer", "peripheral", "device", "object"]);
        self.add_chain(&["software", "program", "technology"]);
        self.add_chain(&["application", "software", "program", "technology"]);
        self.add_chain(&["database", "software", "technology"]);

        // Emotions/States
        self.add_chain(&["happiness", "positive_emotion", "emotion", "mental_state"]);
        self.add_chain(&["joy", "happiness", "positive_emotion", "emotion"]);
        self.add_chain(&["sadness", "negative_emotion", "emotion", "mental_state"]);
        self.add_chain(&["anger", "negative_emotion", "emotion", "mental_state"]);
        self.add_chain(&["fear", "negative_emotion", "emotion", "mental_state"]);
        self.add_chain(&["love", "positive_emotion", "emotion", "mental_state"]);

        // Actions
        self.add_chain(&["running", "moving", "physical_activity", "action"]);
        self.add_chain(&["walking", "moving", "physical_activity", "action"]);
        self.add_chain(&["swimming", "moving", "physical_activity", "action"]);
        self.add_chain(&["eating", "consuming", "action"]);
        self.add_chain(&["drinking", "consuming", "action"]);
        self.add_chain(&["reading", "learning", "mental_activity", "action"]);
        self.add_chain(&["writing", "creating", "mental_activity", "action"]);
        self.add_chain(&["thinking", "mental_activity", "action"]);
        self.add_chain(&["sleeping", "resting", "action"]);
        self.add_chain(&["working", "action"]);

        // Time concepts
        self.add_chain(&["monday", "weekday", "day", "time_unit"]);
        self.add_chain(&["tuesday", "weekday", "day", "time_unit"]);
        self.add_chain(&["wednesday", "weekday", "day", "time_unit"]);
        self.add_chain(&["thursday", "weekday", "day", "time_unit"]);
        self.add_chain(&["friday", "weekday", "day", "time_unit"]);
        self.add_chain(&["saturday", "weekend", "day", "time_unit"]);
        self.add_chain(&["sunday", "weekend", "day", "time_unit"]);
        self.add_chain(&["january", "month", "time_unit"]);
        self.add_chain(&["morning", "time_of_day", "time_unit"]);
        self.add_chain(&["afternoon", "time_of_day", "time_unit"]);
        self.add_chain(&["evening", "time_of_day", "time_unit"]);
        self.add_chain(&["night", "time_of_day", "time_unit"]);
    }

    /// Add a chain of IS-A relations: a is-a b is-a c is-a ...
    pub fn add_chain(&mut self, chain: &[&str]) {
        for i in 0..chain.len().saturating_sub(1) {
            self.add_relation(chain[i], chain[i + 1]);
        }
    }

    /// Add single IS-A relation: child is-a parent
    pub fn add_relation(&mut self, child: &str, parent: &str) {
        let child_key = child.to_lowercase();
        let parent_key = parent.to_lowercase();

        // Update child node
        let child_node = self
            .nodes
            .entry(child_key.clone())
            .or_insert_with(|| ConceptNode {
                parents: Vec::new(),
                children: Vec::new(),
            });
        if !child_node.parents.contains(&parent_key) {
            child_node.parents.push(parent_key.clone());
        }

        // Update parent node
        let parent_node = self.nodes.entry(parent_key).or_insert_with(|| ConceptNode {
            parents: Vec::new(),
            children: Vec::new(),
        });
        if !parent_node.children.contains(&child_key) {
            parent_node.children.push(child_key);
        }
    }

    /// Get parent concepts (what this IS-A)
    pub fn get_parents(&self, concept: &str) -> Vec<&str> {
        self.nodes
            .get(&concept.to_lowercase())
            .map(|n| n.parents.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get child concepts (what IS-A this)
    pub fn get_children(&self, concept: &str) -> Vec<&str> {
        self.nodes
            .get(&concept.to_lowercase())
            .map(|n| n.children.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all ancestors (transitive closure of parents)
    pub fn get_ancestors(&self, concept: &str, max_depth: usize) -> Vec<String> {
        let mut ancestors = Vec::new();
        let mut visited = HashSet::new();
        self.collect_ancestors(
            &concept.to_lowercase(),
            &mut ancestors,
            &mut visited,
            0,
            max_depth,
        );
        ancestors
    }

    /// Get all descendants (transitive closure of children)
    pub fn get_descendants(&self, concept: &str, max_depth: usize) -> Vec<String> {
        let mut descendants = Vec::new();
        let mut visited = HashSet::new();
        self.collect_descendants(
            &concept.to_lowercase(),
            &mut descendants,
            &mut visited,
            0,
            max_depth,
        );
        descendants
    }

    fn collect_ancestors(
        &self,
        concept: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
        depth: usize,
        max_depth: usize,
    ) {
        if depth >= max_depth || visited.contains(concept) {
            return;
        }
        visited.insert(concept.to_string());

        if let Some(node) = self.nodes.get(concept) {
            for parent in &node.parents {
                if !visited.contains(parent) {
                    result.push(parent.clone());
                    self.collect_ancestors(parent, result, visited, depth + 1, max_depth);
                }
            }
        }
    }

    fn collect_descendants(
        &self,
        concept: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
        depth: usize,
        max_depth: usize,
    ) {
        if depth >= max_depth || visited.contains(concept) {
            return;
        }
        visited.insert(concept.to_string());

        if let Some(node) = self.nodes.get(concept) {
            for child in &node.children {
                if !visited.contains(child) {
                    result.push(child.clone());
                    self.collect_descendants(child, result, visited, depth + 1, max_depth);
                }
            }
        }
    }

    /// Expand query upward through concept hierarchy
    ///
    /// Query "dog" also matches "animal", "mammal", "living_thing"
    /// Returns expanded text string (caller generates SDR)
    pub fn expand_upward(&self, text: &str, max_depth: usize) -> String {
        let mut expanded_text = String::new();

        for word in text.split_whitespace() {
            expanded_text.push_str(word);
            expanded_text.push(' ');

            // Add all ancestors
            for ancestor in self.get_ancestors(word, max_depth) {
                expanded_text.push_str(&ancestor);
                expanded_text.push(' ');
            }
        }

        expanded_text.trim().to_string()
    }

    /// Expand query downward through concept hierarchy
    ///
    /// Query "animal" also matches "dog", "cat", "bird"
    /// Returns expanded text string (caller generates SDR)
    pub fn expand_downward(&self, text: &str, max_depth: usize) -> String {
        let mut expanded_text = String::new();

        for word in text.split_whitespace() {
            expanded_text.push_str(word);
            expanded_text.push(' ');

            // Add all descendants
            for descendant in self.get_descendants(word, max_depth) {
                expanded_text.push_str(&descendant);
                expanded_text.push(' ');
            }
        }

        expanded_text.trim().to_string()
    }

    /// Check if child IS-A parent (directly or transitively)
    pub fn is_a(&self, child: &str, parent: &str, max_depth: usize) -> bool {
        let child_lower = child.to_lowercase();
        let parent_lower = parent.to_lowercase();

        if child_lower == parent_lower {
            return true;
        }

        self.get_ancestors(&child_lower, max_depth)
            .contains(&parent_lower)
    }

    /// Get number of concepts
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for ConceptGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_relation() {
        let graph = ConceptGraph::new();

        let parents = graph.get_parents("dog");
        assert!(parents.contains(&"mammal"));
    }

    #[test]
    fn test_ancestors() {
        let graph = ConceptGraph::new();

        let ancestors = graph.get_ancestors("dog", 5);
        assert!(ancestors.contains(&"mammal".to_string()));
        assert!(ancestors.contains(&"animal".to_string()));
        assert!(ancestors.contains(&"living_thing".to_string()));
    }

    #[test]
    fn test_descendants() {
        let graph = ConceptGraph::new();

        let descendants = graph.get_descendants("animal", 3);
        assert!(descendants.contains(&"mammal".to_string()));
        assert!(
            descendants.contains(&"dog".to_string()) || descendants.contains(&"cat".to_string())
        );
    }

    #[test]
    fn test_is_a() {
        let graph = ConceptGraph::new();

        assert!(graph.is_a("dog", "animal", 5));
        assert!(graph.is_a("dog", "living_thing", 5));
        assert!(!graph.is_a("dog", "vehicle", 5));
    }

    #[test]
    fn test_custom_relation() {
        let mut graph = ConceptGraph::empty();
        graph.add_relation("aura_memory", "software");
        graph.add_relation("software", "technology");

        assert!(graph.is_a("aura_memory", "software", 3));
        assert!(graph.is_a("aura_memory", "technology", 3));
    }

    #[test]
    fn test_graph_size() {
        let graph = ConceptGraph::new();
        assert!(graph.len() > 50, "Should have at least 50 concepts");
    }
}
