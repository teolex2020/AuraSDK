/**
 * Aura C FFI — Cognitive Memory for AI Agents
 *
 * Build: cargo build --release --no-default-features --features "encryption,ffi"
 * Output: target/release/aura.dll | libaura.so | libaura.dylib
 *
 * Memory rules:
 *   - Strings returned by aura_* must be freed with aura_free_string()
 *   - AuraHandle must be freed with aura_close() then aura_free()
 */

#ifndef AURA_H
#define AURA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef void* AuraHandle;

/* Memory levels */
#define AURA_LEVEL_DEFAULT   0
#define AURA_LEVEL_WORKING   1
#define AURA_LEVEL_DECISIONS 2
#define AURA_LEVEL_DOMAIN    3
#define AURA_LEVEL_IDENTITY  4

/* ── Lifecycle ── */

AuraHandle aura_open(const char* path, char** out_error);
AuraHandle aura_open_encrypted(const char* path, const char* password, char** out_error);
int        aura_close(AuraHandle handle, char** out_error);
void       aura_free(AuraHandle handle);
void       aura_free_string(char* s);

/* ── Core operations ── */

/* Store: returns record ID (caller frees). level=0 for default. */
char* aura_store(AuraHandle handle, const char* content, uint8_t level,
                 const char* tags_json, const char* namespace_, char** out_error);

/* Recall as formatted text. token_budget=0 for default. */
char* aura_recall(AuraHandle handle, const char* query,
                  int32_t token_budget, char** out_error);

/* Recall structured: returns JSON array. top_k=0 for default. */
char* aura_recall_structured(AuraHandle handle, const char* query,
                             int32_t top_k, char** out_error);

/* ── Maintenance ── */

int aura_run_maintenance(AuraHandle handle, char** out_error);

/* ── Stats ── */

int64_t aura_count(AuraHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* AURA_H */
