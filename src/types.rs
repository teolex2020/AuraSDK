use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Dynamic energy of a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pulse {
    pub intensity: f32,
    pub stability: f32, // Added for Hybrid Learning
    pub decay_velocity: f32,
    pub last_resonance: f64, // Unix timestamp
}

impl Default for Pulse {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            stability: 1.0, // Default stability
            decay_velocity: 0.0,
            last_resonance: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        }
    }
}

/// Level of variability and uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flux {
    pub entropy: f32,
    pub parent_id: Option<String>,
    pub dna: String, // "user_core", "general", "hieroglyph"
}

impl Default for Flux {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            parent_id: None,
            dna: "general".to_string(),
        }
    }
}

/// Synthetic synapse - a unit of living memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuraSynapse {
    pub id: String,
    pub text: String,
    pub sdr_indices: Vec<u16>, // Using u16 for 8192 bits (0-8191)
    pub pulse: Pulse,
    pub flux: Flux,
}

impl AuraSynapse {
    pub fn new(id: String, text: String, sdr_indices: Vec<u16>) -> Self {
        Self {
            id,
            text,
            sdr_indices,
            pulse: Pulse::default(),
            flux: Flux::default(),
        }
    }
}
