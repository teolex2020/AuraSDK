//! Multi-Tenancy Module for Aura Memory
//!
//! Provides enterprise-grade tenant isolation:
//! - Namespace isolation (tenant_id prefix)
//! - Per-tenant encryption keys
//! - Quota management (memory/record limits)
//! - Tenant-aware operations

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::crypto::EncryptionKey;
use crate::memory::AuraMemory;

/// Tenant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Unique tenant identifier
    pub id: String,
    /// Human-readable tenant name
    pub name: String,
    /// Maximum number of records allowed (0 = unlimited)
    pub max_records: u64,
    /// Maximum storage size in bytes (0 = unlimited)
    pub max_storage_bytes: u64,
    /// Whether encryption is required for this tenant
    pub require_encryption: bool,
    /// Tenant metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (Unix ms)
    pub created_at: u64,
    /// Last access timestamp (Unix ms)
    pub last_access: u64,
}

impl TenantConfig {
    /// Create a new tenant configuration
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: id.into(),
            name: name.into(),
            max_records: 0,       // Unlimited
            max_storage_bytes: 0, // Unlimited
            require_encryption: false,
            metadata: HashMap::new(),
            created_at: now,
            last_access: now,
        }
    }

    /// Create tenant with quotas
    pub fn with_quotas(
        id: impl Into<String>,
        name: impl Into<String>,
        max_records: u64,
        max_storage_bytes: u64,
    ) -> Self {
        let mut config = Self::new(id, name);
        config.max_records = max_records;
        config.max_storage_bytes = max_storage_bytes;
        config
    }

    /// Require encryption for this tenant
    pub fn require_encryption(mut self) -> Self {
        self.require_encryption = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Tenant usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantUsage {
    /// Current record count
    pub record_count: u64,
    /// Current storage size in bytes
    pub storage_bytes: u64,
    /// Total operations performed
    pub total_operations: u64,
    /// Store operations
    pub store_ops: u64,
    /// Retrieve operations
    pub retrieve_ops: u64,
    /// Last operation timestamp
    pub last_operation: u64,
}

impl TenantUsage {
    /// Check if tenant has exceeded record quota
    pub fn exceeds_record_quota(&self, config: &TenantConfig) -> bool {
        config.max_records > 0 && self.record_count >= config.max_records
    }

    /// Check if tenant has exceeded storage quota
    pub fn exceeds_storage_quota(&self, config: &TenantConfig) -> bool {
        config.max_storage_bytes > 0 && self.storage_bytes >= config.max_storage_bytes
    }

    /// Increment operation counters
    pub fn record_store_op(&mut self) {
        self.store_ops += 1;
        self.total_operations += 1;
        self.last_operation = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Increment retrieve counter
    pub fn record_retrieve_op(&mut self) {
        self.retrieve_ops += 1;
        self.total_operations += 1;
        self.last_operation = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }
}

/// Multi-tenant memory manager
pub struct TenantManager {
    /// Base directory for all tenant data
    base_path: PathBuf,
    /// Loaded tenant configurations
    tenants: RwLock<HashMap<String, TenantConfig>>,
    /// Tenant usage tracking
    usage: RwLock<HashMap<String, TenantUsage>>,
    /// Cached tenant memory instances
    instances: RwLock<HashMap<String, Arc<AuraMemory>>>,
    /// Master encryption key (for tenant key derivation)
    master_key: Option<EncryptionKey>,
}

impl TenantManager {
    /// Create a new tenant manager
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;

        let manager = Self {
            base_path,
            tenants: RwLock::new(HashMap::new()),
            usage: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            master_key: None,
        };

        // Load existing tenants
        manager.load_tenants()?;

        Ok(manager)
    }

    /// Create tenant manager with master encryption key
    pub fn with_encryption<P: AsRef<Path>>(
        base_path: P,
        master_key: EncryptionKey,
    ) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;

        let manager = Self {
            base_path,
            tenants: RwLock::new(HashMap::new()),
            usage: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            master_key: Some(master_key),
        };

        manager.load_tenants()?;

        Ok(manager)
    }

    /// Load existing tenant configurations from disk
    fn load_tenants(&self) -> Result<()> {
        let tenants_file = self.base_path.join("tenants.json");

        if tenants_file.exists() {
            let content = fs::read_to_string(&tenants_file)?;
            let tenants: HashMap<String, TenantConfig> = serde_json::from_str(&content)?;

            let mut guard = self.tenants.write();
            *guard = tenants;
        }

        // Load usage data
        let usage_file = self.base_path.join("usage.json");
        if usage_file.exists() {
            let content = fs::read_to_string(&usage_file)?;
            let usage: HashMap<String, TenantUsage> = serde_json::from_str(&content)?;

            let mut guard = self.usage.write();
            *guard = usage;
        }

        Ok(())
    }

    /// Save tenant configurations to disk
    fn save_tenants(&self) -> Result<()> {
        let tenants_file = self.base_path.join("tenants.json");
        let guard = self.tenants.read();
        let content = serde_json::to_string_pretty(&*guard)?;
        fs::write(&tenants_file, content)?;
        Ok(())
    }

    /// Save usage data to disk
    fn save_usage(&self) -> Result<()> {
        let usage_file = self.base_path.join("usage.json");
        let guard = self.usage.read();
        let content = serde_json::to_string_pretty(&*guard)?;
        fs::write(&usage_file, content)?;
        Ok(())
    }

    /// Create a new tenant
    pub fn create_tenant(&self, config: TenantConfig) -> Result<()> {
        let tenant_id = config.id.clone();

        // Check if tenant already exists
        {
            let guard = self.tenants.read();
            if guard.contains_key(&tenant_id) {
                return Err(anyhow!("Tenant '{}' already exists", tenant_id));
            }
        }

        // Validate encryption requirement
        if config.require_encryption && self.master_key.is_none() {
            return Err(anyhow!(
                "Tenant requires encryption but no master key configured"
            ));
        }

        // Create tenant directory
        let tenant_dir = self.tenant_path(&tenant_id);
        fs::create_dir_all(&tenant_dir)?;

        // Store configuration
        {
            let mut guard = self.tenants.write();
            guard.insert(tenant_id.clone(), config);
        }

        // Initialize usage
        {
            let mut guard = self.usage.write();
            guard.insert(tenant_id, TenantUsage::default());
        }

        self.save_tenants()?;
        self.save_usage()?;

        Ok(())
    }

    /// Delete a tenant and all its data
    pub fn delete_tenant(&self, tenant_id: &str) -> Result<()> {
        // Remove from memory
        {
            let mut guard = self.instances.write();
            guard.remove(tenant_id);
        }

        // Remove configuration
        {
            let mut guard = self.tenants.write();
            guard.remove(tenant_id);
        }

        // Remove usage
        {
            let mut guard = self.usage.write();
            guard.remove(tenant_id);
        }

        // Delete directory
        let tenant_dir = self.tenant_path(tenant_id);
        if tenant_dir.exists() {
            fs::remove_dir_all(&tenant_dir)?;
        }

        self.save_tenants()?;
        self.save_usage()?;

        Ok(())
    }

    /// Get tenant configuration
    pub fn get_tenant(&self, tenant_id: &str) -> Option<TenantConfig> {
        let guard = self.tenants.read();
        guard.get(tenant_id).cloned()
    }

    /// List all tenants
    pub fn list_tenants(&self) -> Vec<TenantConfig> {
        let guard = self.tenants.read();
        guard.values().cloned().collect()
    }

    /// Get tenant usage statistics
    pub fn get_usage(&self, tenant_id: &str) -> Option<TenantUsage> {
        let guard = self.usage.read();
        guard.get(tenant_id).cloned()
    }

    /// Get or create AuraMemory instance for tenant
    pub fn get_memory(&self, tenant_id: &str) -> Result<Arc<AuraMemory>> {
        // Check cache first
        {
            let guard = self.instances.read();
            if let Some(memory) = guard.get(tenant_id) {
                return Ok(memory.clone());
            }
        }

        // Load tenant config
        let config = self
            .get_tenant(tenant_id)
            .ok_or_else(|| anyhow!("Tenant '{}' not found", tenant_id))?;

        // Check quota before loading
        if let Some(usage) = self.get_usage(tenant_id) {
            if usage.exceeds_record_quota(&config) {
                return Err(anyhow!("Tenant '{}' has exceeded record quota", tenant_id));
            }
            if usage.exceeds_storage_quota(&config) {
                return Err(anyhow!("Tenant '{}' has exceeded storage quota", tenant_id));
            }
        }

        // Create memory instance
        let tenant_dir = self.tenant_path(tenant_id);
        let memory = if config.require_encryption {
            // Derive tenant-specific key from master key
            let tenant_password = self.derive_tenant_password(tenant_id)?;
            AuraMemory::encrypted(&tenant_dir, &tenant_password)?
        } else {
            AuraMemory::new(&tenant_dir)?
        };

        let memory = Arc::new(memory);

        // Cache instance
        {
            let mut guard = self.instances.write();
            guard.insert(tenant_id.to_string(), memory.clone());
        }

        // Update last access
        {
            let mut guard = self.tenants.write();
            if let Some(config) = guard.get_mut(tenant_id) {
                config.last_access = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
            }
        }

        Ok(memory)
    }

    /// Get tenant directory path
    fn tenant_path(&self, tenant_id: &str) -> PathBuf {
        self.base_path.join(format!("tenant_{}", tenant_id))
    }

    /// Derive tenant-specific password from master key
    fn derive_tenant_password(&self, tenant_id: &str) -> Result<String> {
        let master_key = self
            .master_key
            .as_ref()
            .ok_or_else(|| anyhow!("No master key configured"))?;

        // Use HMAC to derive tenant-specific password
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let mut mac: HmacSha256 = Mac::new_from_slice(master_key.as_bytes())
            .map_err(|e| anyhow!("HMAC init failed: {}", e))?;

        mac.update(tenant_id.as_bytes());
        mac.update(b"aura_tenant_key_derivation_v1");

        let result = mac.finalize();
        let bytes = result.into_bytes();

        // Convert to hex string as password
        Ok(hex::encode(&bytes[..]))
    }

    /// Record a store operation for quota tracking
    pub fn record_store(&self, tenant_id: &str, bytes_added: u64) -> Result<()> {
        // Get config for quota check
        let config = self
            .get_tenant(tenant_id)
            .ok_or_else(|| anyhow!("Tenant not found"))?;

        let mut guard = self.usage.write();
        let usage = guard.entry(tenant_id.to_string()).or_default();

        // Check quotas before recording
        if usage.exceeds_record_quota(&config) {
            return Err(anyhow!("Record quota exceeded"));
        }
        if config.max_storage_bytes > 0
            && usage.storage_bytes + bytes_added > config.max_storage_bytes
        {
            return Err(anyhow!("Storage quota exceeded"));
        }

        usage.record_count += 1;
        usage.storage_bytes += bytes_added;
        usage.record_store_op();

        drop(guard);
        self.save_usage()?;

        Ok(())
    }

    /// Record a retrieve operation
    pub fn record_retrieve(&self, tenant_id: &str) -> Result<()> {
        let mut guard = self.usage.write();
        let usage = guard.entry(tenant_id.to_string()).or_default();

        usage.record_retrieve_op();

        drop(guard);
        self.save_usage()?;

        Ok(())
    }

    /// Flush all cached instances
    pub fn flush_all(&self) -> Result<()> {
        let guard = self.instances.read();
        for (_tenant_id, memory) in guard.iter() {
            memory.flush()?;
        }
        Ok(())
    }

    /// Close a specific tenant (remove from cache)
    pub fn close_tenant(&self, tenant_id: &str) {
        let mut guard = self.instances.write();
        if let Some(memory) = guard.remove(tenant_id) {
            let _ = memory.flush();
        }
    }

    /// Get total statistics across all tenants
    pub fn get_total_usage(&self) -> TenantUsage {
        let guard = self.usage.read();
        let mut total = TenantUsage::default();

        for usage in guard.values() {
            total.record_count += usage.record_count;
            total.storage_bytes += usage.storage_bytes;
            total.total_operations += usage.total_operations;
            total.store_ops += usage.store_ops;
            total.retrieve_ops += usage.retrieve_ops;
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_tenant() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        let config = TenantConfig::new("acme", "Acme Corporation");
        manager.create_tenant(config).unwrap();

        let loaded = manager.get_tenant("acme").unwrap();
        assert_eq!(loaded.name, "Acme Corporation");
    }

    #[test]
    fn test_tenant_isolation() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        // Create two tenants
        manager
            .create_tenant(TenantConfig::new("tenant_a", "Tenant A"))
            .unwrap();
        manager
            .create_tenant(TenantConfig::new("tenant_b", "Tenant B"))
            .unwrap();

        // Get separate memory instances
        let mem_a = manager.get_memory("tenant_a").unwrap();
        let mem_b = manager.get_memory("tenant_b").unwrap();

        // Store in A
        mem_a.process("Secret data for A", false).unwrap();
        mem_a.flush().unwrap();

        // Search in B should find nothing
        let results = mem_b.retrieve("Secret data", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_tenant_quotas() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        let config = TenantConfig::with_quotas("limited", "Limited Tenant", 2, 0);
        manager.create_tenant(config).unwrap();

        // Record stores up to limit
        manager.record_store("limited", 100).unwrap();
        manager.record_store("limited", 100).unwrap();

        // Third should fail
        let result = manager.record_store("limited", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_tenants() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        manager
            .create_tenant(TenantConfig::new("a", "Tenant A"))
            .unwrap();
        manager
            .create_tenant(TenantConfig::new("b", "Tenant B"))
            .unwrap();
        manager
            .create_tenant(TenantConfig::new("c", "Tenant C"))
            .unwrap();

        let tenants = manager.list_tenants();
        assert_eq!(tenants.len(), 3);
    }

    #[test]
    fn test_delete_tenant() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        manager
            .create_tenant(TenantConfig::new("delete_me", "To Delete"))
            .unwrap();
        assert!(manager.get_tenant("delete_me").is_some());

        manager.delete_tenant("delete_me").unwrap();
        assert!(manager.get_tenant("delete_me").is_none());
    }

    #[test]
    fn test_usage_tracking() {
        let dir = tempdir().unwrap();
        let manager = TenantManager::new(dir.path()).unwrap();

        manager
            .create_tenant(TenantConfig::new("tracked", "Tracked"))
            .unwrap();

        manager.record_store("tracked", 500).unwrap();
        manager.record_store("tracked", 500).unwrap();
        manager.record_retrieve("tracked").unwrap();

        let usage = manager.get_usage("tracked").unwrap();
        assert_eq!(usage.record_count, 2);
        assert_eq!(usage.storage_bytes, 1000);
        assert_eq!(usage.store_ops, 2);
        assert_eq!(usage.retrieve_ops, 1);
    }

    #[test]
    fn test_encrypted_tenant() {
        let dir = tempdir().unwrap();
        let master_key = EncryptionKey::generate();
        let manager = TenantManager::with_encryption(dir.path(), master_key).unwrap();

        let config = TenantConfig::new("secure", "Secure Tenant").require_encryption();
        manager.create_tenant(config).unwrap();

        let memory = manager.get_memory("secure").unwrap();
        assert!(memory.is_encrypted());
    }
}
