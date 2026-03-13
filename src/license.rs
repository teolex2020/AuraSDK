//! Universal Shield License Enforcement
//!
//! Verifies distinct license files signed by the Administrator's Private Key.
//! This allows a single binary to serve multiple clients with different restrictions.
//!
//! Mechanism:
//! 1. Load `license.lic` (JSON: payload + signature)
//! 2. Verify signature using embedded PRODUCT_PUBLIC_KEY
//! 3. Decode payload (HWID, Expiry, Client)
//! 4. Check HWID against local hardware
//! 5. Check Expiry against current date

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{NaiveDate, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;

// =============================================================================
// CRYPTO CONFIGURATION (Injected at build time by build_shark.ps1)
// =============================================================================

/// Public Key for verifying licenses (Hex String)
/// Value "{{PUBLIC_KEY}}" means development mode (no check)
const PRODUCT_PUBLIC_KEY_HEX: &str = "{{PUBLIC_KEY}}";

// =============================================================================
// DATA STRUCTURES
// =============================================================================

#[derive(Debug, Deserialize)]
struct LicenseFile {
    payload: String,   // Base64 encoded JSON
    signature: String, // Base64 encoded Ed25519 signature
    #[allow(dead_code)]
    alg: String,
}

#[derive(Debug, Deserialize)]
struct LicensePayload {
    client: String,
    hwid: String,
    expiry: String, // ISO 8601
    #[allow(dead_code)]
    issued: String,
}

// =============================================================================
// HARDWARE IDENTIFICATION
// =============================================================================

pub fn get_system_id() -> String {
    let mut components: Vec<String> = Vec::new();

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("wmic")
            .args(["cpu", "get", "processorid"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && trimmed != "ProcessorId" {
                    components.push(format!("CPU:{}", trimmed));
                    break;
                }
            }
        }
        if let Ok(output) = Command::new("wmic")
            .args(["baseboard", "get", "serialnumber"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && trimmed != "SerialNumber" {
                    components.push(format!("MB:{}", trimmed));
                    break;
                }
            }
        }
        if let Ok(output) = Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("MachineGuid") {
                    if let Some(guid) = line.split_whitespace().last() {
                        components.push(format!("GUID:{}", guid));
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(machine_id) = fs::read_to_string("/etc/machine-id") {
            components.push(format!("MID:{}", machine_id.trim()));
        }
    }

    if components.is_empty() {
        if let Ok(hostname) = env::var("COMPUTERNAME").or_else(|_| env::var("HOSTNAME")) {
            components.push(format!("HOST:{}", hostname));
        }
    }

    let combined = components.join("|");
    let mut hasher = Sha256::new();
    hasher.update(combined.as_bytes());
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// VERIFICATION LOGIC
// =============================================================================

pub fn enforce_license() {
    // 1. Check if Dev Mode
    if PRODUCT_PUBLIC_KEY_HEX.starts_with("{{") {
        return; // Dev mode
    }

    // 2. Load License File
    let license_path = env::var("AURA_LICENSE_PATH").unwrap_or_else(|_| "license.lic".to_string());
    if !Path::new(&license_path).exists() {
        die(
            "LICENSE FILE MISSING",
            &format!("Could not find '{}'.", license_path),
        );
    }

    let license_content = match fs::read_to_string(&license_path) {
        Ok(c) => c,
        Err(e) => die("READ ERROR", &e.to_string()),
    };

    let license_file: LicenseFile = match serde_json::from_str(&license_content) {
        Ok(l) => l,
        Err(_) => die("INVALID FORMAT", "License file is corrupted or not JSON."),
    };

    // 3. Verify Signature
    let pub_key_bytes = match hex::decode(PRODUCT_PUBLIC_KEY_HEX) {
        Ok(b) => b,
        Err(_) => die("INTERNAL ERROR", "Embedded public key is invalid."),
    };

    let pub_key_arr: [u8; 32] = match pub_key_bytes.as_slice().try_into() {
        Ok(arr) => arr,
        Err(_) => die("INTERNAL ERROR", "Embedded public key must be 32 bytes."),
    };
    let verifying_key = match VerifyingKey::from_bytes(&pub_key_arr) {
        Ok(k) => k,
        Err(_) => die("INTERNAL ERROR", "Embedded public key format error."),
    };

    let sig_bytes = match BASE64.decode(&license_file.signature) {
        Ok(b) => b,
        Err(_) => die("INVALID SIGNATURE", "Signature encoding error."),
    };

    let sig_arr: [u8; 64] = match sig_bytes.as_slice().try_into() {
        Ok(arr) => arr,
        Err(_) => die(
            "INVALID SIGNATURE",
            "Signature length invalid (must be 64 bytes).",
        ),
    };
    let signature = Signature::from_bytes(&sig_arr);

    // 4. Decode Payload (We sign the raw JSON bytes)
    let json_bytes = match BASE64.decode(&license_file.payload) {
        Ok(b) => b,
        Err(_) => die("PAYLOAD ERROR", "Could not decode license payload."),
    };

    if verifying_key.verify(&json_bytes, &signature).is_err() {
        die(
            "SECURITY ALERT",
            "License signature verification failed!\nThis file has been tampered with.",
        );
    }

    let payload: LicensePayload = match serde_json::from_slice(&json_bytes) {
        Ok(p) => p,
        Err(_) => die("PAYLOAD ERROR", "Invalid license data structure."),
    };

    // 5. Check Requirements
    let current_hwid = get_system_id();
    if payload.hwid != current_hwid {
        die(
            "UNAUTHORIZED HARDWARE",
            &format!(
                "License Host: {}...\nSystem Host:  {}...\nClient: {}",
                &payload.hwid[..12],
                &current_hwid[..12],
                payload.client
            ),
        );
    }

    if payload.expiry != "PERMANENT" {
        let expiry_date = match NaiveDate::parse_from_str(&payload.expiry, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => match datetime_iso(&payload.expiry) {
                Some(d) => d,
                None => die("DATE ERROR", "Invalid expiry date format."),
            },
        };

        let today = Utc::now().date_naive();
        if today > expiry_date {
            die(
                "LICENSE EXPIRED",
                &format!("Expired on: {}\nClient: {}", payload.expiry, payload.client),
            );
        }

        let days = (expiry_date - today).num_days();
        if days <= 7 {
            eprintln!("[WARNING] License expires in {} days.", days);
        }
    }
}

fn datetime_iso(s: &str) -> Option<NaiveDate> {
    // Handle ISO format if full datetime provided
    s.split('T')
        .next()
        .and_then(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d").ok())
}

fn die(title: &str, msg: &str) -> ! {
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║ {:^56} ║", title);
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    for line in msg.lines() {
        eprintln!("║  {:<56}║", line);
    }
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    std::process::exit(1);
}

// Helper for other modules
pub fn get_license_info() -> LicenseInfo {
    LicenseInfo {
        hardware_bound: !PRODUCT_PUBLIC_KEY_HEX.starts_with("{{"),
        is_development: PRODUCT_PUBLIC_KEY_HEX.starts_with("{{"),
    }
}

pub struct LicenseInfo {
    pub hardware_bound: bool,
    pub is_development: bool,
}
