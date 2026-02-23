//! Cryptographic module for Aura Memory
//!
//! Provides:
//! - AES-256-GCM / ChaCha20-Poly1305 encryption at rest
//! - Argon2id key derivation from passwords
//! - HMAC-SHA256 for data integrity
//! - Secure memory handling with zeroize

use anyhow::{anyhow, Result};
use argon2::Argon2;
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce, Key,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::ZeroizeOnDrop;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

/// HMAC-SHA256 type alias
type HmacSha256 = Hmac<Sha256>;

/// Encryption key with secure memory cleanup
#[derive(Clone, ZeroizeOnDrop)]
pub struct EncryptionKey {
    key: [u8; 32],
}

impl EncryptionKey {
    /// Create a new random encryption key
    pub fn generate() -> Self {
        let key = ChaCha20Poly1305::generate_key(&mut OsRng);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(key.as_slice());
        Self { key: key_bytes }
    }

    /// Derive key from password using Argon2id
    pub fn from_password(password: &str, salt: &[u8; 16]) -> Result<Self> {
        let argon2 = Argon2::default();

        // Use raw hash for key derivation
        let mut key = [0u8; 32];
        argon2.hash_password_into(
            password.as_bytes(),
            salt,
            &mut key,
        ).map_err(|e| anyhow!("Key derivation failed: {}", e))?;

        Ok(Self { key })
    }

    /// Get the raw key bytes (use carefully!)
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    /// Export key to file (encrypted with master password)
    pub fn save_to_file(&self, path: &Path, master_password: &str) -> Result<()> {
        let salt = generate_salt();
        let master_key = Self::from_password(master_password, &salt)?;

        let encrypted = encrypt_data(&self.key, &master_key)?;

        let mut file = File::create(path)?;
        file.write_all(&salt)?;
        file.write_all(&encrypted)?;

        Ok(())
    }

    /// Load key from file (decrypted with master password)
    pub fn load_from_file(path: &Path, master_password: &str) -> Result<Self> {
        let mut file = File::open(path)?;

        let mut salt = [0u8; 16];
        file.read_exact(&mut salt)?;

        let mut encrypted = Vec::new();
        file.read_to_end(&mut encrypted)?;

        let master_key = Self::from_password(master_password, &salt)?;
        let decrypted = decrypt_data(&encrypted, &master_key)?;

        if decrypted.len() != 32 {
            return Err(anyhow!("Invalid key length"));
        }

        let mut key = [0u8; 32];
        key.copy_from_slice(&decrypted);

        Ok(Self { key })
    }
}

/// Generate a random 16-byte salt
pub fn generate_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    use rand::RngCore;
    OsRng.fill_bytes(&mut salt);
    salt
}

/// Generate a random 12-byte nonce for ChaCha20-Poly1305
pub fn generate_nonce() -> [u8; 12] {
    let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(nonce.as_slice());
    nonce_bytes
}

/// Encrypt data using ChaCha20-Poly1305
/// Returns: nonce (12 bytes) + ciphertext (variable) + tag (16 bytes)
pub fn encrypt_data(plaintext: &[u8], key: &EncryptionKey) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key.key));
    let nonce_bytes = generate_nonce();
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow!("Encryption failed: {}", e))?;

    // Prepend nonce to ciphertext
    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);

    Ok(result)
}

/// Decrypt data using ChaCha20-Poly1305
/// Input: nonce (12 bytes) + ciphertext (variable) + tag (16 bytes)
pub fn decrypt_data(encrypted: &[u8], key: &EncryptionKey) -> Result<Vec<u8>> {
    if encrypted.len() < 12 + 16 {
        return Err(anyhow!("Encrypted data too short"));
    }

    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key.key));
    let nonce = Nonce::from_slice(&encrypted[..12]);
    let ciphertext = &encrypted[12..];

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow!("Decryption failed - invalid key or corrupted data"))?;

    Ok(plaintext)
}

/// Compute HMAC-SHA256 for data integrity
pub fn compute_hmac(data: &[u8], key: &EncryptionKey) -> [u8; 32] {
    let mut mac: HmacSha256 = Mac::new_from_slice(&key.key)
        .expect("HMAC can take key of any size");
    mac.update(data);
    let result = mac.finalize();

    let mut hmac_bytes = [0u8; 32];
    hmac_bytes.copy_from_slice(&result.into_bytes());
    hmac_bytes
}

/// Verify HMAC-SHA256
pub fn verify_hmac(data: &[u8], expected_hmac: &[u8; 32], key: &EncryptionKey) -> bool {
    let mut mac: HmacSha256 = Mac::new_from_slice(&key.key)
        .expect("HMAC can take key of any size");
    mac.update(data);
    mac.verify_slice(expected_hmac).is_ok()
}

/// Encrypted file wrapper for brain.aura
pub struct EncryptedStorage {
    key: EncryptionKey,
    path: std::path::PathBuf,
}

impl EncryptedStorage {
    /// Create new encrypted storage
    pub fn new(path: impl AsRef<Path>, key: EncryptionKey) -> Self {
        Self {
            key,
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Create with password-derived key
    pub fn with_password(path: impl AsRef<Path>, password: &str) -> Result<Self> {
        let key_path = path.as_ref().with_extension("key");

        let key = if key_path.exists() {
            // Load existing key
            EncryptionKey::load_from_file(&key_path, password)?
        } else {
            // Generate new key and save it
            let key = EncryptionKey::generate();
            key.save_to_file(&key_path, password)?;
            key
        };

        Ok(Self::new(path, key))
    }

    /// Write encrypted data to file
    pub fn write(&self, data: &[u8]) -> Result<()> {
        let encrypted = encrypt_data(data, &self.key)?;
        let hmac = compute_hmac(&encrypted, &self.key);

        let mut file = File::create(&self.path)?;

        // Write: HMAC (32) + encrypted data
        file.write_all(&hmac)?;
        file.write_all(&encrypted)?;
        file.sync_all()?;

        Ok(())
    }

    /// Read and decrypt data from file
    pub fn read(&self) -> Result<Vec<u8>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(&self.path)?;

        let mut hmac = [0u8; 32];
        file.read_exact(&mut hmac)?;

        let mut encrypted = Vec::new();
        file.read_to_end(&mut encrypted)?;

        // Verify integrity
        if !verify_hmac(&encrypted, &hmac, &self.key) {
            return Err(anyhow!("Data integrity check failed - file may be corrupted or tampered"));
        }

        decrypt_data(&encrypted, &self.key)
    }

    /// Append encrypted record (for append-only storage)
    pub fn append_record(&self, record: &[u8]) -> Result<u64> {
        let encrypted = encrypt_data(record, &self.key)?;
        let hmac = compute_hmac(&encrypted, &self.key);

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let offset = file.metadata()?.len();

        // Record format: length (4 bytes) + HMAC (32) + encrypted data
        let total_len = (32 + encrypted.len()) as u32;
        file.write_all(&total_len.to_le_bytes())?;
        file.write_all(&hmac)?;
        file.write_all(&encrypted)?;

        Ok(offset)
    }

    /// Read encrypted record at offset
    pub fn read_record(&self, offset: u64) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;

        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(offset))?;

        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        let total_len = u32::from_le_bytes(len_bytes) as usize;

        if total_len < 32 {
            return Err(anyhow!("Invalid record length"));
        }

        let mut hmac = [0u8; 32];
        file.read_exact(&mut hmac)?;

        let mut encrypted = vec![0u8; total_len - 32];
        file.read_exact(&mut encrypted)?;

        // Verify integrity
        if !verify_hmac(&encrypted, &hmac, &self.key) {
            return Err(anyhow!("Record integrity check failed"));
        }

        decrypt_data(&encrypted, &self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_key_generation() {
        let key = EncryptionKey::generate();
        assert_eq!(key.as_bytes().len(), 32);
    }

    #[test]
    fn test_key_from_password() {
        let salt = generate_salt();
        let key1 = EncryptionKey::from_password("test_password", &salt).unwrap();
        let key2 = EncryptionKey::from_password("test_password", &salt).unwrap();

        // Same password + salt = same key
        assert_eq!(key1.as_bytes(), key2.as_bytes());

        // Different password = different key
        let key3 = EncryptionKey::from_password("other_password", &salt).unwrap();
        assert_ne!(key1.as_bytes(), key3.as_bytes());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let key = EncryptionKey::generate();
        let plaintext = b"Hello, Aura Memory!";

        let encrypted = encrypt_data(plaintext, &key).unwrap();
        assert_ne!(encrypted.as_slice(), plaintext);

        let decrypted = decrypt_data(&encrypted, &key).unwrap();
        assert_eq!(decrypted.as_slice(), plaintext);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = EncryptionKey::generate();
        let key2 = EncryptionKey::generate();
        let plaintext = b"Secret data";

        let encrypted = encrypt_data(plaintext, &key1).unwrap();
        let result = decrypt_data(&encrypted, &key2);

        assert!(result.is_err());
    }

    #[test]
    fn test_hmac_verification() {
        let key = EncryptionKey::generate();
        let data = b"Important data";

        let hmac = compute_hmac(data, &key);
        assert!(verify_hmac(data, &hmac, &key));

        // Tampered data should fail
        let tampered = b"Modified data";
        assert!(!verify_hmac(tampered, &hmac, &key));
    }

    #[test]
    fn test_encrypted_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.enc");

        let key = EncryptionKey::generate();
        let storage = EncryptedStorage::new(&path, key.clone());

        let data = b"Sensitive memory data";
        storage.write(data).unwrap();

        let read_data = storage.read().unwrap();
        assert_eq!(read_data.as_slice(), data);
    }

    #[test]
    fn test_key_file_save_load() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("master.key");

        let original_key = EncryptionKey::generate();
        original_key.save_to_file(&key_path, "master_password").unwrap();

        let loaded_key = EncryptionKey::load_from_file(&key_path, "master_password").unwrap();
        assert_eq!(original_key.as_bytes(), loaded_key.as_bytes());

        // Wrong password should fail
        let result = EncryptionKey::load_from_file(&key_path, "wrong_password");
        assert!(result.is_err());
    }

    #[test]
    fn test_append_read_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("records.enc");

        let key = EncryptionKey::generate();
        let storage = EncryptedStorage::new(&path, key);

        let record1 = b"First record";
        let record2 = b"Second record";

        let offset1 = storage.append_record(record1).unwrap();
        let offset2 = storage.append_record(record2).unwrap();

        assert_eq!(offset1, 0);
        assert!(offset2 > offset1);

        let read1 = storage.read_record(offset1).unwrap();
        let read2 = storage.read_record(offset2).unwrap();

        assert_eq!(read1.as_slice(), record1);
        assert_eq!(read2.as_slice(), record2);
    }
}
