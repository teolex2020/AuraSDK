//! User profile and agent persona schemas.
//!
//! Rewritten from brain_tools.py identity logic.

use std::collections::HashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ── User Profile ──

/// Standard user profile fields.
pub const PROFILE_FIELDS: &[&str] = &[
    "name", "full_name", "age", "birthday", "gender",
    "location", "city", "country", "timezone",
    "language", "languages", "occupation", "company",
    "email", "phone", "bio",
];

/// Format profile data as human-readable content string.
pub fn format_profile_content(fields: &HashMap<String, String>) -> String {
    let mut lines = vec!["User Profile:".to_string()];
    for &key in PROFILE_FIELDS {
        if let Some(val) = fields.get(key) {
            if !val.is_empty() {
                lines.push(format!("  {}: {}", key, val));
            }
        }
    }
    // Include any extra fields not in standard set
    for (key, val) in fields {
        if !PROFILE_FIELDS.contains(&key.as_str()) && !val.is_empty() {
            lines.push(format!("  {}: {}", key, val));
        }
    }
    lines.join("\n")
}

// ── Agent Persona ──

/// Agent persona trait values (0.0-1.0).
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersonaTraits {
    pub warmth: f32,
    pub curiosity: f32,
    pub conciseness: f32,
    pub humor: f32,
    pub formality: f32,
}

impl Default for PersonaTraits {
    fn default() -> Self {
        Self {
            warmth: 0.8,
            curiosity: 0.7,
            conciseness: 0.6,
            humor: 0.4,
            formality: 0.4,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PersonaTraits {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

/// Agent persona configuration.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentPersona {
    pub name: String,
    pub role: String,
    pub scope: String,
    pub tone: String,
    pub motivations: String,
    pub languages: Vec<String>,
    pub catchphrases: Vec<String>,
    pub avoid: Vec<String>,
    pub traits: PersonaTraits,
}

impl Default for AgentPersona {
    fn default() -> Self {
        Self {
            name: "Assistant".into(),
            role: "personal assistant".into(),
            scope: "universal".into(),
            tone: "warm and friendly".into(),
            motivations: "help the user accomplish their goals".into(),
            languages: vec!["en".into()],
            catchphrases: vec![],
            avoid: vec![],
            traits: PersonaTraits::default(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AgentPersona {
    #[new]
    fn py_new() -> Self {
        Self::default()
    }
}

/// Convert persona to system instruction text.
pub fn persona_to_instruction(persona: &AgentPersona) -> String {
    let mut lines = vec![
        format!("You are {}, a {}.", persona.name, persona.role),
        format!("You help users with anything they need — {}.", persona.motivations),
        format!("Your scope: {}.", persona.scope),
        format!("Speak {}.", persona.tone),
    ];

    if !persona.catchphrases.is_empty() {
        lines.push(format!(
            "Signature phrases you may use: {}.",
            persona.catchphrases.join(", ")
        ));
    }

    if !persona.avoid.is_empty() {
        lines.push(format!("Avoid: {}.", persona.avoid.join(", ")));
    }

    let mut trait_hints = Vec::new();
    if persona.traits.humor > 0.6 {
        trait_hints.push("feel free to use light humor");
    }
    if persona.traits.formality > 0.7 {
        trait_hints.push("maintain a professional tone");
    }
    if persona.traits.conciseness > 0.8 {
        trait_hints.push("be especially concise");
    }
    if !trait_hints.is_empty() {
        lines.push(format!("Style: {}.", trait_hints.join("; ")));
    }

    lines.join(" ") + "\n\n"
}

/// The tag used for persona records in brain storage.
pub const PERSONA_TAG: &str = "agent-persona";

/// The tag used for user profile records in brain storage.
pub const PROFILE_TAG: &str = "user-profile";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_profile() {
        let mut fields = HashMap::new();
        fields.insert("name".into(), "Teo".into());
        fields.insert("age".into(), "25".into());
        fields.insert("city".into(), "Kyiv".into());

        let content = format_profile_content(&fields);
        assert!(content.contains("name: Teo"));
        assert!(content.contains("age: 25"));
        assert!(content.contains("city: Kyiv"));
    }

    #[test]
    fn test_persona_to_instruction() {
        let persona = AgentPersona {
            name: "Remy".into(),
            role: "personal assistant".into(),
            tone: "warm and friendly".into(),
            traits: PersonaTraits {
                humor: 0.8,
                formality: 0.3,
                ..Default::default()
            },
            ..Default::default()
        };

        let instruction = persona_to_instruction(&persona);
        assert!(instruction.contains("You are Remy"));
        assert!(instruction.contains("light humor"));
        assert!(!instruction.contains("professional tone")); // formality < 0.7
    }

    #[test]
    fn test_default_persona() {
        let persona = AgentPersona::default();
        assert_eq!(persona.name, "Assistant");
        assert_eq!(persona.traits.warmth, 0.8);
    }
}
