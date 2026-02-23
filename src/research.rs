//! Research orchestrator — multi-step research as a first-class SDK feature.
//!
//! Rewritten from brain_tools.py research logic.

use std::collections::HashMap;
use parking_lot::RwLock;
use uuid::Uuid;

use crate::credibility::SourceCredibility;

/// Research project status.
#[derive(Debug, Clone, PartialEq)]
pub enum ResearchStatus {
    Active,
    Completed,
    Cancelled,
}

/// A research finding — one data point in a research project.
#[derive(Debug, Clone)]
pub struct ResearchFinding {
    pub query: String,
    pub result: String,
    pub url: Option<String>,
    pub credibility: f32,
    pub timestamp: String,
}

/// A research project — tracks queries, findings, and synthesis.
#[derive(Debug, Clone)]
pub struct ResearchProject {
    pub id: String,
    pub topic: String,
    pub depth: String,
    pub status: ResearchStatus,
    pub queries: Vec<String>,
    pub findings: Vec<ResearchFinding>,
    pub synthesis: Option<String>,
    pub created_at: String,
}

/// Research orchestrator managing multiple research projects.
pub struct ResearchEngine {
    /// Active projects: project_id → project.
    projects: RwLock<HashMap<String, ResearchProject>>,
    /// Source credibility scorer.
    credibility: RwLock<SourceCredibility>,
}

impl ResearchEngine {
    pub fn new() -> Self {
        Self {
            projects: RwLock::new(HashMap::new()),
            credibility: RwLock::new(SourceCredibility::new()),
        }
    }

    /// Start a new research project.
    ///
    /// Returns the project with suggested queries (placeholder — real queries
    /// come from LLM via `llm_fn` callback if available).
    pub fn start_research(&self, topic: &str, depth: Option<&str>) -> ResearchProject {
        let depth = depth.unwrap_or("standard");
        let num_queries = match depth {
            "quick" => 2,
            "deep" => 7,
            _ => 4, // standard
        };

        // Generate placeholder queries — in real usage, LLM generates these
        let queries: Vec<String> = (0..num_queries)
            .map(|i| format!("{} query {}", topic, i + 1))
            .collect();

        let project = ResearchProject {
            id: Uuid::new_v4().to_string(),
            topic: topic.to_string(),
            depth: depth.to_string(),
            status: ResearchStatus::Active,
            queries,
            findings: Vec::new(),
            synthesis: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let id = project.id.clone();
        self.projects.write().insert(id, project.clone());
        project
    }

    /// Add a research finding to a project.
    pub fn add_finding(
        &self,
        project_id: &str,
        query: &str,
        result: &str,
        url: Option<&str>,
    ) -> Result<(), String> {
        let credibility_score = url
            .map(|u| self.credibility.read().get_score(u))
            .unwrap_or(0.5);

        let finding = ResearchFinding {
            query: query.to_string(),
            result: result.to_string(),
            url: url.map(|s| s.to_string()),
            credibility: credibility_score,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let mut projects = self.projects.write();
        let project = projects.get_mut(project_id)
            .ok_or_else(|| format!("Project {} not found", project_id))?;

        if project.status != ResearchStatus::Active {
            return Err("Project is not active".to_string());
        }

        project.findings.push(finding);
        Ok(())
    }

    /// Complete a research project. Returns the project for storage.
    ///
    /// If `synthesis` is provided (from LLM), it's attached to the project.
    /// Without synthesis, the project is completed with raw findings only.
    pub fn complete_research(
        &self,
        project_id: &str,
        synthesis: Option<String>,
    ) -> Result<ResearchProject, String> {
        let mut projects = self.projects.write();
        let project = projects.get_mut(project_id)
            .ok_or_else(|| format!("Project {} not found", project_id))?;

        project.status = ResearchStatus::Completed;
        project.synthesis = synthesis;

        Ok(project.clone())
    }

    /// Cancel a research project.
    pub fn cancel_research(&self, project_id: &str) -> Result<(), String> {
        let mut projects = self.projects.write();
        let project = projects.get_mut(project_id)
            .ok_or_else(|| format!("Project {} not found", project_id))?;
        project.status = ResearchStatus::Cancelled;
        Ok(())
    }

    /// Get all active research projects.
    pub fn active_projects(&self) -> Vec<ResearchProject> {
        self.projects.read()
            .values()
            .filter(|p| p.status == ResearchStatus::Active)
            .cloned()
            .collect()
    }

    /// Get a specific project by ID.
    pub fn get_project(&self, project_id: &str) -> Option<ResearchProject> {
        self.projects.read().get(project_id).cloned()
    }

    /// Set a credibility override for a domain.
    pub fn set_credibility_override(&self, domain: &str, score: f32) {
        self.credibility.write().set_override(domain, score);
    }

    /// Get credibility score for a URL.
    pub fn get_credibility(&self, url: &str) -> f32 {
        self.credibility.read().get_score(url)
    }
}

impl Default for ResearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_lifecycle() {
        let engine = ResearchEngine::new();

        // Start
        let project = engine.start_research("GRPO for memory ranking", Some("standard"));
        assert_eq!(project.status, ResearchStatus::Active);
        assert_eq!(project.queries.len(), 4);

        // Add findings
        engine.add_finding(
            &project.id, "GRPO paper", "Published by DeepSeek...",
            Some("https://arxiv.org/paper/123"),
        ).unwrap();

        let p = engine.get_project(&project.id).unwrap();
        assert_eq!(p.findings.len(), 1);
        assert!(p.findings[0].credibility > 0.8); // arxiv = 0.90

        // Complete
        let completed = engine.complete_research(
            &project.id, Some("GRPO is a group-relative policy optimization...".into()),
        ).unwrap();
        assert_eq!(completed.status, ResearchStatus::Completed);
        assert!(completed.synthesis.is_some());
    }

    #[test]
    fn test_credibility_in_findings() {
        let engine = ResearchEngine::new();
        let project = engine.start_research("test", None);

        engine.add_finding(&project.id, "q1", "result", Some("https://reddit.com/r/test")).unwrap();
        engine.add_finding(&project.id, "q2", "result", Some("https://nature.com/123")).unwrap();

        let p = engine.get_project(&project.id).unwrap();
        assert!(p.findings[1].credibility > p.findings[0].credibility);
    }
}
