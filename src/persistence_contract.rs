use std::collections::BTreeMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub const PERSISTENCE_MANIFEST_FILE: &str = "persistence_manifest.json";
pub const PERSISTENCE_SCHEMA_VERSION: u32 = 1;
pub const BELIEF_STORE_VERSION: u32 = 1;
pub const CONCEPT_STORE_VERSION: u32 = 1;
pub const CAUSAL_STORE_VERSION: u32 = 1;
pub const POLICY_STORE_VERSION: u32 = 1;
pub const MAINTENANCE_TRENDS_VERSION: u32 = 1;
pub const REFLECTION_SUMMARIES_VERSION: u32 = 1;

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PersistenceManifest {
    pub schema_version: u32,
    pub surfaces: BTreeMap<String, u32>,
}

impl PersistenceManifest {
    pub fn current() -> Self {
        let mut surfaces = BTreeMap::new();
        surfaces.insert("belief".into(), BELIEF_STORE_VERSION);
        surfaces.insert("concept".into(), CONCEPT_STORE_VERSION);
        surfaces.insert("causal".into(), CAUSAL_STORE_VERSION);
        surfaces.insert("policy".into(), POLICY_STORE_VERSION);
        surfaces.insert("maintenance_trends".into(), MAINTENANCE_TRENDS_VERSION);
        surfaces.insert("reflection_summaries".into(), REFLECTION_SUMMARIES_VERSION);
        Self {
            schema_version: PERSISTENCE_SCHEMA_VERSION,
            surfaces,
        }
    }
}
