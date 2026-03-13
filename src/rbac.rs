//! Role-Based Access Control (RBAC) for Aura Memory
//!
//! Provides enterprise-grade access control:
//! - Role definitions (Admin/Writer/Reader)
//! - API key authentication
//! - Per-namespace permissions
//! - Token validation with expiry

use anyhow::{anyhow, Result};
use hmac::{Hmac, Mac};
use parking_lot::RwLock;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Available roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Full access - create/delete tenants, manage users, all operations
    Admin,
    /// Read and write memories, but no admin operations
    Writer,
    /// Read-only access to memories
    Reader,
}

impl Role {
    /// Check if role can perform an action
    pub fn can_perform(&self, action: &Action) -> bool {
        match self {
            Role::Admin => true, // Admin can do everything
            Role::Writer => matches!(
                action,
                Action::Read | Action::Write | Action::Delete | Action::Search
            ),
            Role::Reader => matches!(action, Action::Read | Action::Search),
        }
    }

    /// Get role name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Admin => "admin",
            Role::Writer => "writer",
            Role::Reader => "reader",
        }
    }
}

/// Actions that can be controlled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Read/retrieve memories
    Read,
    /// Store new memories
    Write,
    /// Delete memories
    Delete,
    /// Search/query memories
    Search,
    /// Create tenants
    CreateTenant,
    /// Delete tenants
    DeleteTenant,
    /// Manage API keys
    ManageKeys,
    /// Access audit logs
    ViewAudit,
    /// Backup operations
    Backup,
    /// Restore operations
    Restore,
}

/// API key with permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique key ID (public)
    pub id: String,
    /// The actual key (hashed for storage)
    pub key_hash: String,
    /// User or service name
    pub name: String,
    /// Assigned role
    pub role: Role,
    /// Tenant restrictions (empty = all tenants)
    pub allowed_tenants: HashSet<String>,
    /// Specific namespace restrictions
    pub allowed_namespaces: HashSet<String>,
    /// Creation timestamp
    pub created_at: u64,
    /// Expiry timestamp (0 = never expires)
    pub expires_at: u64,
    /// Whether key is currently active
    pub active: bool,
    /// Last used timestamp
    pub last_used: u64,
    /// Usage count
    pub usage_count: u64,
}

impl ApiKey {
    /// Check if key is expired
    pub fn is_expired(&self) -> bool {
        if self.expires_at == 0 {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now > self.expires_at
    }

    /// Check if key is valid (active and not expired)
    pub fn is_valid(&self) -> bool {
        self.active && !self.is_expired()
    }

    /// Check if key can access a tenant
    pub fn can_access_tenant(&self, tenant_id: &str) -> bool {
        self.allowed_tenants.is_empty() || self.allowed_tenants.contains(tenant_id)
    }

    /// Check if key can access a namespace
    pub fn can_access_namespace(&self, namespace: &str) -> bool {
        self.allowed_namespaces.is_empty() || self.allowed_namespaces.contains(namespace)
    }

    /// Check if key can perform an action
    pub fn can_perform(&self, action: &Action) -> bool {
        self.is_valid() && self.role.can_perform(action)
    }
}

/// Result of authentication attempt
#[derive(Debug)]
pub struct AuthResult {
    /// Whether authentication succeeded
    pub success: bool,
    /// The authenticated key (if success)
    pub key: Option<ApiKey>,
    /// Error message (if failure)
    pub error: Option<String>,
}

impl AuthResult {
    fn success(key: ApiKey) -> Self {
        Self {
            success: true,
            key: Some(key),
            error: None,
        }
    }

    fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            key: None,
            error: Some(error.into()),
        }
    }
}

/// RBAC Manager
pub struct AccessControl {
    /// Path to store keys
    storage_path: PathBuf,
    /// Secret for HMAC signing
    secret: [u8; 32],
    /// Cached API keys
    keys: RwLock<HashMap<String, ApiKey>>,
}

impl AccessControl {
    /// Create a new access control manager
    pub fn new<P: AsRef<Path>>(storage_path: P) -> Result<Self> {
        let storage_path = storage_path.as_ref().to_path_buf();
        fs::create_dir_all(&storage_path)?;

        // Load or generate secret
        let secret_path = storage_path.join("rbac_secret.key");
        let secret = if secret_path.exists() {
            let data = fs::read(&secret_path)?;
            if data.len() != 32 {
                return Err(anyhow!("Invalid secret file"));
            }
            let mut secret = [0u8; 32];
            secret.copy_from_slice(&data);
            secret
        } else {
            let mut secret = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut secret);
            fs::write(&secret_path, secret)?;
            secret
        };

        let manager = Self {
            storage_path,
            secret,
            keys: RwLock::new(HashMap::new()),
        };

        manager.load_keys()?;

        Ok(manager)
    }

    /// Load keys from disk
    fn load_keys(&self) -> Result<()> {
        let keys_file = self.storage_path.join("api_keys.json");

        if keys_file.exists() {
            let content = fs::read_to_string(&keys_file)?;
            let keys: HashMap<String, ApiKey> = serde_json::from_str(&content)?;

            let mut guard = self.keys.write();
            *guard = keys;
        }

        Ok(())
    }

    /// Save keys to disk
    fn save_keys(&self) -> Result<()> {
        let keys_file = self.storage_path.join("api_keys.json");
        let guard = self.keys.read();
        let content = serde_json::to_string_pretty(&*guard)?;
        fs::write(&keys_file, content)?;
        Ok(())
    }

    /// Generate a new API key
    pub fn create_key(
        &self,
        name: impl Into<String>,
        role: Role,
        expires_in_days: Option<u32>,
    ) -> Result<(String, String)> {
        // Generate random key
        let mut key_bytes = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut key_bytes);

        // Create key ID and secret
        let key_id = format!("aura_{}", &hex::encode(&key_bytes[..8]));
        let key_secret = hex::encode(key_bytes);
        let full_key = format!("{}_{}", key_id, key_secret);

        // Hash the key for storage
        let key_hash = self.hash_key(&full_key);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let expires_at = if let Some(days) = expires_in_days {
            now + (days as u64 * 24 * 60 * 60 * 1000)
        } else {
            0 // Never expires
        };

        let api_key = ApiKey {
            id: key_id.clone(),
            key_hash,
            name: name.into(),
            role,
            allowed_tenants: HashSet::new(),
            allowed_namespaces: HashSet::new(),
            created_at: now,
            expires_at,
            active: true,
            last_used: 0,
            usage_count: 0,
        };

        // Store key
        {
            let mut guard = self.keys.write();
            guard.insert(key_id.clone(), api_key);
        }

        self.save_keys()?;

        // Return the full key (only time it's visible)
        Ok((key_id, full_key))
    }

    /// Create key with tenant restrictions
    pub fn create_key_for_tenant(
        &self,
        name: impl Into<String>,
        role: Role,
        tenant_id: impl Into<String>,
        expires_in_days: Option<u32>,
    ) -> Result<(String, String)> {
        let (key_id, full_key) = self.create_key(name, role, expires_in_days)?;

        // Add tenant restriction
        {
            let mut guard = self.keys.write();
            if let Some(key) = guard.get_mut(&key_id) {
                key.allowed_tenants.insert(tenant_id.into());
            }
        }

        self.save_keys()?;

        Ok((key_id, full_key))
    }

    /// Hash a key using HMAC-SHA256
    fn hash_key(&self, key: &str) -> String {
        let mut mac: HmacSha256 =
            Mac::new_from_slice(&self.secret).expect("HMAC can take key of any size");
        mac.update(key.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    /// Authenticate with an API key (Bearer token)
    pub fn authenticate(&self, bearer_token: &str) -> AuthResult {
        // Remove "Bearer " prefix if present
        let key = bearer_token.strip_prefix("Bearer ").unwrap_or(bearer_token);

        // Extract key ID from the key
        let parts: Vec<&str> = key.splitn(2, '_').collect();
        if parts.len() < 2 {
            return AuthResult::failure("Invalid key format");
        }

        let key_id = format!(
            "{}_{}",
            parts[0],
            parts[1].chars().take(16).collect::<String>()
        );

        // Lookup key
        let guard = self.keys.read();
        let api_key = match guard.get(&key_id) {
            Some(k) => k.clone(),
            None => return AuthResult::failure("Key not found"),
        };
        drop(guard);

        // Verify hash
        let expected_hash = self.hash_key(key);
        if api_key.key_hash != expected_hash {
            return AuthResult::failure("Invalid key");
        }

        // Check validity
        if !api_key.active {
            return AuthResult::failure("Key is deactivated");
        }

        if api_key.is_expired() {
            return AuthResult::failure("Key has expired");
        }

        // Update usage stats
        {
            let mut guard = self.keys.write();
            if let Some(k) = guard.get_mut(&key_id) {
                k.last_used = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                k.usage_count += 1;
            }
        }

        AuthResult::success(api_key)
    }

    /// Check authorization for an action
    pub fn authorize(&self, key: &ApiKey, action: Action, tenant_id: Option<&str>) -> Result<()> {
        // Check role permission
        if !key.role.can_perform(&action) {
            return Err(anyhow!(
                "Role '{}' cannot perform action '{:?}'",
                key.role.as_str(),
                action
            ));
        }

        // Check tenant access
        if let Some(tenant) = tenant_id {
            if !key.can_access_tenant(tenant) {
                return Err(anyhow!("Access denied to tenant '{}'", tenant));
            }
        }

        Ok(())
    }

    /// Revoke (deactivate) a key
    pub fn revoke_key(&self, key_id: &str) -> Result<()> {
        let mut guard = self.keys.write();
        match guard.get_mut(key_id) {
            Some(key) => {
                key.active = false;
                drop(guard);
                self.save_keys()?;
                Ok(())
            }
            None => Err(anyhow!("Key not found")),
        }
    }

    /// Delete a key permanently
    pub fn delete_key(&self, key_id: &str) -> Result<()> {
        let mut guard = self.keys.write();
        if guard.remove(key_id).is_none() {
            return Err(anyhow!("Key not found"));
        }
        drop(guard);
        self.save_keys()?;
        Ok(())
    }

    /// List all keys (without secrets)
    pub fn list_keys(&self) -> Vec<ApiKey> {
        let guard = self.keys.read();
        guard.values().cloned().collect()
    }

    /// Get key by ID
    pub fn get_key(&self, key_id: &str) -> Option<ApiKey> {
        let guard = self.keys.read();
        guard.get(key_id).cloned()
    }

    /// Update key role
    pub fn update_role(&self, key_id: &str, new_role: Role) -> Result<()> {
        let mut guard = self.keys.write();
        match guard.get_mut(key_id) {
            Some(key) => {
                key.role = new_role;
                drop(guard);
                self.save_keys()?;
                Ok(())
            }
            None => Err(anyhow!("Key not found")),
        }
    }

    /// Add tenant access to a key
    pub fn add_tenant_access(&self, key_id: &str, tenant_id: &str) -> Result<()> {
        let mut guard = self.keys.write();
        match guard.get_mut(key_id) {
            Some(key) => {
                key.allowed_tenants.insert(tenant_id.to_string());
                drop(guard);
                self.save_keys()?;
                Ok(())
            }
            None => Err(anyhow!("Key not found")),
        }
    }

    /// Remove tenant access from a key
    pub fn remove_tenant_access(&self, key_id: &str, tenant_id: &str) -> Result<()> {
        let mut guard = self.keys.write();
        match guard.get_mut(key_id) {
            Some(key) => {
                key.allowed_tenants.remove(tenant_id);
                drop(guard);
                self.save_keys()?;
                Ok(())
            }
            None => Err(anyhow!("Key not found")),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> RbacStats {
        let guard = self.keys.read();

        let total = guard.len();
        let active = guard.values().filter(|k| k.active).count();
        let expired = guard.values().filter(|k| k.is_expired()).count();

        let by_role = guard.values().fold(HashMap::new(), |mut acc, k| {
            *acc.entry(k.role.as_str().to_string()).or_insert(0) += 1;
            acc
        });

        RbacStats {
            total_keys: total,
            active_keys: active,
            expired_keys: expired,
            keys_by_role: by_role,
        }
    }
}

/// RBAC statistics
#[derive(Debug, Clone, Serialize)]
pub struct RbacStats {
    pub total_keys: usize,
    pub active_keys: usize,
    pub expired_keys: usize,
    pub keys_by_role: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_authenticate() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        let (_key_id, full_key) = ac.create_key("Test User", Role::Writer, None).unwrap();

        // Authenticate
        let result = ac.authenticate(&full_key);
        assert!(result.success);
        assert_eq!(result.key.unwrap().role, Role::Writer);

        // Wrong key should fail
        let result = ac.authenticate("invalid_key");
        assert!(!result.success);
    }

    #[test]
    fn test_role_permissions() {
        // Admin can do everything
        assert!(Role::Admin.can_perform(&Action::ManageKeys));
        assert!(Role::Admin.can_perform(&Action::CreateTenant));
        assert!(Role::Admin.can_perform(&Action::Write));

        // Writer can read/write but not admin
        assert!(Role::Writer.can_perform(&Action::Write));
        assert!(Role::Writer.can_perform(&Action::Read));
        assert!(!Role::Writer.can_perform(&Action::ManageKeys));

        // Reader can only read/search
        assert!(Role::Reader.can_perform(&Action::Read));
        assert!(Role::Reader.can_perform(&Action::Search));
        assert!(!Role::Reader.can_perform(&Action::Write));
        assert!(!Role::Reader.can_perform(&Action::Delete));
    }

    #[test]
    fn test_tenant_restriction() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        let (_key_id, full_key) = ac
            .create_key_for_tenant("Tenant User", Role::Writer, "acme", None)
            .unwrap();

        let result = ac.authenticate(&full_key);
        assert!(result.success);

        let key = result.key.unwrap();
        assert!(key.can_access_tenant("acme"));
        assert!(!key.can_access_tenant("other"));
    }

    #[test]
    fn test_key_expiry() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        // Create key that expires in 0 days (already expired)
        let (key_id, _) = ac.create_key("Expiring", Role::Reader, Some(0)).unwrap();

        // Manually set to expired
        {
            let mut guard = ac.keys.write();
            if let Some(key) = guard.get_mut(&key_id) {
                key.expires_at = 1; // Expired in 1970
            }
        }

        let key = ac.get_key(&key_id).unwrap();
        assert!(key.is_expired());
        assert!(!key.is_valid());
    }

    #[test]
    fn test_revoke_key() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        let (key_id, full_key) = ac.create_key("To Revoke", Role::Writer, None).unwrap();

        // Should work initially
        let result = ac.authenticate(&full_key);
        assert!(result.success);

        // Revoke
        ac.revoke_key(&key_id).unwrap();

        // Should fail now
        let result = ac.authenticate(&full_key);
        assert!(!result.success);
        assert_eq!(result.error.unwrap(), "Key is deactivated");
    }

    #[test]
    fn test_authorization() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        let (_, full_key) = ac
            .create_key_for_tenant("Limited", Role::Reader, "allowed_tenant", None)
            .unwrap();

        let result = ac.authenticate(&full_key);
        let key = result.key.unwrap();

        // Should succeed
        ac.authorize(&key, Action::Read, Some("allowed_tenant"))
            .unwrap();

        // Should fail - wrong action
        assert!(ac
            .authorize(&key, Action::Write, Some("allowed_tenant"))
            .is_err());

        // Should fail - wrong tenant
        assert!(ac
            .authorize(&key, Action::Read, Some("other_tenant"))
            .is_err());
    }

    #[test]
    fn test_stats() {
        let dir = tempdir().unwrap();
        let ac = AccessControl::new(dir.path()).unwrap();

        ac.create_key("Admin", Role::Admin, None).unwrap();
        ac.create_key("Writer 1", Role::Writer, None).unwrap();
        ac.create_key("Writer 2", Role::Writer, None).unwrap();
        ac.create_key("Reader", Role::Reader, None).unwrap();

        let stats = ac.stats();
        assert_eq!(stats.total_keys, 4);
        assert_eq!(stats.active_keys, 4);
        assert_eq!(*stats.keys_by_role.get("admin").unwrap_or(&0), 1);
        assert_eq!(*stats.keys_by_role.get("writer").unwrap_or(&0), 2);
        assert_eq!(*stats.keys_by_role.get("reader").unwrap_or(&0), 1);
    }
}
