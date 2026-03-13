//! C-compatible FFI bindings for Aura.
//!
//! Provides an opaque-pointer API for Go, C#, Java, and other languages
//! that can call C shared libraries.
//!
//! # Build
//! ```bash
//! cargo build --release --no-default-features --features "encryption,ffi"
//! ```
//!
//! # Memory rules
//! - Strings returned by `aura_*` functions must be freed with `aura_free_string`.
//! - The `AuraHandle` must be freed with `aura_close` then `aura_free`.
//! - `out_error` parameters are set on failure; caller frees with `aura_free_string`.

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use crate::aura::Aura;
use crate::levels::Level;

/// Opaque handle to an Aura instance.
pub type AuraHandle = *mut Aura;

// ── Helpers ──

fn set_error(out_error: *mut *mut c_char, msg: &str) {
    if !out_error.is_null() {
        if let Ok(c) = CString::new(msg) {
            unsafe { *out_error = c.into_raw() };
        }
    }
}

fn cstr_to_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p).to_str().ok() }
}

fn level_from_u8(v: u8) -> Option<Level> {
    match v {
        1 => Some(Level::Working),
        2 => Some(Level::Decisions),
        3 => Some(Level::Domain),
        4 => Some(Level::Identity),
        _ => None,
    }
}

// ── Lifecycle ──

/// Open (or create) an Aura brain at the given path.
///
/// Returns an opaque handle, or NULL on error (check `out_error`).
#[no_mangle]
pub extern "C" fn aura_open(path: *const c_char, out_error: *mut *mut c_char) -> AuraHandle {
    let path_str = match cstr_to_str(path) {
        Some(s) => s,
        None => {
            set_error(out_error, "path is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };
    match Aura::open(path_str) {
        Ok(a) => Box::into_raw(Box::new(a)),
        Err(e) => {
            set_error(out_error, &e.to_string());
            ptr::null_mut()
        }
    }
}

/// Open an encrypted Aura brain.
#[no_mangle]
pub extern "C" fn aura_open_encrypted(
    path: *const c_char,
    password: *const c_char,
    out_error: *mut *mut c_char,
) -> AuraHandle {
    let path_str = match cstr_to_str(path) {
        Some(s) => s,
        None => {
            set_error(out_error, "path is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };
    let pwd = match cstr_to_str(password) {
        Some(s) => s,
        None => {
            set_error(out_error, "password is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };
    match Aura::open_with_password(path_str, Some(pwd)) {
        Ok(a) => Box::into_raw(Box::new(a)),
        Err(e) => {
            set_error(out_error, &e.to_string());
            ptr::null_mut()
        }
    }
}

/// Close and flush the Aura instance. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn aura_close(handle: AuraHandle, out_error: *mut *mut c_char) -> i32 {
    if handle.is_null() {
        set_error(out_error, "handle is null");
        return -1;
    }
    let aura = unsafe { &*handle };
    match aura.close() {
        Ok(_) => 0,
        Err(e) => {
            set_error(out_error, &e.to_string());
            -1
        }
    }
}

/// Free the Aura handle. Call after aura_close.
#[no_mangle]
pub extern "C" fn aura_free(handle: AuraHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Free a string returned by any aura_* function.
#[no_mangle]
pub extern "C" fn aura_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

// ── Store ──

/// Store a memory. Returns the record ID as a C string (caller frees).
///
/// - `level`: 1=Working, 2=Decisions, 3=Domain, 4=Identity. 0 = default (Working).
/// - `tags_json`: JSON array of strings, e.g. `["tag1","tag2"]`. NULL = no tags.
/// - `namespace`: namespace string. NULL = "default".
#[no_mangle]
pub extern "C" fn aura_store(
    handle: AuraHandle,
    content: *const c_char,
    level: u8,
    tags_json: *const c_char,
    namespace: *const c_char,
    out_error: *mut *mut c_char,
) -> *mut c_char {
    if handle.is_null() {
        set_error(out_error, "handle is null");
        return ptr::null_mut();
    }
    let aura = unsafe { &*handle };

    let content_str = match cstr_to_str(content) {
        Some(s) => s,
        None => {
            set_error(out_error, "content is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };

    let lv = if level == 0 {
        None
    } else {
        level_from_u8(level)
    };

    let tags: Option<Vec<String>> =
        cstr_to_str(tags_json).and_then(|s| serde_json::from_str(s).ok());

    let ns = cstr_to_str(namespace);

    match aura.store_with_channel(
        content_str,
        lv,
        tags,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ns,
        None,
    ) {
        Ok(rec) => match CString::new(rec.id.as_str()) {
            Ok(c) => c.into_raw(),
            Err(_) => {
                set_error(out_error, "record ID contains null byte");
                ptr::null_mut()
            }
        },
        Err(e) => {
            set_error(out_error, &e.to_string());
            ptr::null_mut()
        }
    }
}

// ── Recall ──

/// Recall as formatted text (for LLM prompt injection).
///
/// Returns a C string (caller frees). NULL on error.
/// `token_budget`: 0 = default (2048).
#[no_mangle]
pub extern "C" fn aura_recall(
    handle: AuraHandle,
    query: *const c_char,
    token_budget: i32,
    out_error: *mut *mut c_char,
) -> *mut c_char {
    if handle.is_null() {
        set_error(out_error, "handle is null");
        return ptr::null_mut();
    }
    let aura = unsafe { &*handle };

    let query_str = match cstr_to_str(query) {
        Some(s) => s,
        None => {
            set_error(out_error, "query is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };

    let budget = if token_budget > 0 {
        Some(token_budget as usize)
    } else {
        None
    };

    match aura.recall(query_str, budget, None, None, None, None) {
        Ok(text) => match CString::new(text) {
            Ok(c) => c.into_raw(),
            Err(_) => {
                set_error(out_error, "result contains null byte");
                ptr::null_mut()
            }
        },
        Err(e) => {
            set_error(out_error, &e.to_string());
            ptr::null_mut()
        }
    }
}

/// Recall structured results as JSON array.
///
/// Returns JSON: `[{"id":"...","content":"...","score":0.95,"level":"Domain",...}, ...]`
/// `top_k`: 0 = default (20).
#[no_mangle]
pub extern "C" fn aura_recall_structured(
    handle: AuraHandle,
    query: *const c_char,
    top_k: i32,
    out_error: *mut *mut c_char,
) -> *mut c_char {
    if handle.is_null() {
        set_error(out_error, "handle is null");
        return ptr::null_mut();
    }
    let aura = unsafe { &*handle };

    let query_str = match cstr_to_str(query) {
        Some(s) => s,
        None => {
            set_error(out_error, "query is null or invalid UTF-8");
            return ptr::null_mut();
        }
    };

    let k = if top_k > 0 {
        Some(top_k as usize)
    } else {
        None
    };

    match aura.recall_structured(query_str, k, None, None, None, None) {
        Ok(results) => {
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|(score, rec)| {
                    serde_json::json!({
                        "id": rec.id,
                        "content": rec.content,
                        "score": score,
                        "level": rec.level.name(),
                        "strength": rec.strength,
                        "tags": rec.tags,
                        "created_at": rec.created_at,
                        "source_type": rec.source_type,
                    })
                })
                .collect();

            match serde_json::to_string(&json_results) {
                Ok(s) => match CString::new(s) {
                    Ok(c) => c.into_raw(),
                    Err(_) => {
                        set_error(out_error, "JSON contains null byte");
                        ptr::null_mut()
                    }
                },
                Err(e) => {
                    set_error(out_error, &e.to_string());
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            set_error(out_error, &e.to_string());
            ptr::null_mut()
        }
    }
}

// ── Maintenance ──

/// Run a full maintenance cycle. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn aura_run_maintenance(handle: AuraHandle, out_error: *mut *mut c_char) -> i32 {
    if handle.is_null() {
        set_error(out_error, "handle is null");
        return -1;
    }
    let aura = unsafe { &*handle };
    let _ = aura.run_maintenance();
    0
}

// ── Stats ──

/// Get record count. Returns -1 on error.
#[no_mangle]
pub extern "C" fn aura_count(handle: AuraHandle) -> i64 {
    if handle.is_null() {
        return -1;
    }
    let aura = unsafe { &*handle };
    aura.count(None) as i64
}
