#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StartupValidationEvent {
    pub surface: String,
    pub path: String,
    pub status: String,
    pub detail: Option<String>,
    pub recovered: bool,
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StartupValidationReport {
    pub loaded_surfaces: usize,
    pub missing_fallbacks: usize,
    pub recovered_fallbacks: usize,
    pub derived_skips: usize,
    pub has_recovery_warnings: bool,
    pub events: Vec<StartupValidationEvent>,
}
