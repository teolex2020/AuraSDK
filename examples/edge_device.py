"""Edge device memory — IoT, embedded, air-gapped environments.

Demonstrates what makes Aura unique:
- 2.7 MB binary, no external dependencies
- Works completely offline (no LLM, no embeddings API, no cloud)
- Encrypted at rest (ChaCha20-Poly1305 + Argon2id)
- Auto-protect sensitive data (PII detection)
- Circuit breaker for unreliable network tools
- Background maintenance (decay stale data, save disk space)

Use cases:
- Smart home hub with local AI
- Industrial IoT sensor with on-device agent
- Medical device that cannot send data to cloud
- Air-gapped military/government system

Run:
    python examples/edge_device.py
"""

import time
from aura import Aura, Level, MaintenanceConfig, TagTaxonomy

# ── Initialize encrypted brain ──
# All data encrypted at rest. No data leaves the device.
brain = Aura("./edge_data", password="demo-device-key")
print(f"Encrypted brain: {brain.is_encrypted()}")
print(f"Binary size: ~2.7 MB (pure Rust, no Python deps)")

# ── Configure for edge constraints ──
# Protect sensitive data automatically (regex-based PII detection)
tax = TagTaxonomy()
tax.sensitive_tags = {"medical", "credential", "contact", "financial"}
tax.consolidation_skip_tags = {"credential", "contact"}  # never merge sensitive records
brain.set_taxonomy(tax)

# ── Simulate IoT sensor data ──
print("\n--- Storing sensor readings ---")
sensors = [
    ("Temperature: 22.5C, Humidity: 45%, CO2: 412ppm", ["sensor", "environment"]),
    ("Motion detected in Zone A at 14:32", ["sensor", "security"]),
    ("Power consumption: 2.3kW, Solar output: 1.8kW", ["sensor", "energy"]),
    ("User entered code 1234 on keypad", ["sensor", "security"]),
    ("Device firmware v3.2.1 installed successfully", ["system", "update"]),
    ("Network latency to gateway: 45ms, packet loss: 0.1%", ["sensor", "network"]),
    # These will get auto-protected (PII detected by regex):
    ("User's phone number: +380991234567 registered", ["user"]),
    ("Send payment to 0x1234567890abcdef1234567890abcdef12345678", ["transaction"]),
    ("api_key: sk-proj-abc123def456ghi789jkl012mno345pqr", ["config"]),
]

for content, tags in sensors:
    record_id = brain.store(content, tags=tags)
    rec = brain.get(record_id)
    # Auto-protect adds tags like "contact", "credential", "financial"
    auto_tags = [t for t in rec.tags if t not in tags]
    extra = f"  -> auto-protected: {auto_tags}" if auto_tags else ""
    print(f"  Stored: {content[:50]}...{extra}")

# ── Recall without any network/LLM ──
print("\n--- Offline recall (no LLM, no embeddings) ---")
context = brain.recall("security events")
print(context)

print("\n--- Structured recall: energy data ---")
results = brain.recall_structured("power consumption solar", top_k=3)
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} — {r['content'][:60]}")

# ── Circuit breaker for unreliable gateway ──
print("\n--- Circuit breaker: unreliable cloud gateway ---")
# Simulate gateway failures
brain.record_tool_failure("cloud_gateway")
brain.record_tool_failure("cloud_gateway")
brain.record_tool_failure("cloud_gateway")

if brain.is_tool_available("cloud_gateway"):
    print("  Gateway available — syncing data...")
else:
    print("  Gateway UNAVAILABLE — using local-only mode")
    print("  Data stays encrypted on device until gateway recovers")

# Gateway recovers
brain.record_tool_success("cloud_gateway")
print(f"  Gateway recovered: available={brain.is_tool_available('cloud_gateway')}")

# ── Background maintenance (keep disk usage low) ──
print("\n--- Maintenance (decay old readings, save disk) ---")
config = MaintenanceConfig()
config.decay_enabled = True
config.consolidation_enabled = True
brain.configure_maintenance(config)

report = brain.run_maintenance()
print(f"  Records: {report.total_records}")
print(f"  Decayed: {report.decay.decayed}")
print(f"  Archived: {report.records_archived}")

# ── Performance on edge ──
print("\n--- Performance benchmark ---")
start = time.perf_counter()
for i in range(100):
    brain.store(f"Sensor reading #{i}: value={i * 1.5:.1f}", tags=["benchmark"])
store_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
for _ in range(100):
    brain.recall("sensor reading", token_budget=500)
recall_ms = (time.perf_counter() - start) * 1000

print(f"  100 stores: {store_ms:.1f}ms ({store_ms/100:.2f}ms/op)")
print(f"  100 recalls: {recall_ms:.1f}ms ({recall_ms/100:.2f}ms/op)")

stats = brain.stats()
print(f"\n--- Final stats ---")
print(f"  Total records: {stats['total_records']}")
print(f"  Encrypted: {brain.is_encrypted()}")

brain.close()
print("\nEdge brain saved to ./edge_data/ (encrypted)")
