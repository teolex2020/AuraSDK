"""Edge device memory for IoT, embedded, and air-gapped environments.

Demonstrates:
- local operation
- encrypted storage
- sensitive data protection
- resilience when external tools fail
- periodic maintenance

Run:
    python examples/edge_device.py
"""

import time

from aura import Aura, MaintenanceConfig, TagTaxonomy

brain = Aura("./edge_data", password="demo-device-key")
print(f"Encrypted brain: {brain.is_encrypted()}")

tax = TagTaxonomy()
tax.sensitive_tags = {"medical", "credential", "contact", "financial"}
tax.consolidation_skip_tags = {"credential", "contact"}
brain.set_taxonomy(tax)

print("\n--- Storing device events ---")
events = [
    ("Temperature: 22.5C, Humidity: 45%, CO2: 412ppm", ["sensor", "environment"]),
    ("Motion detected in Zone A at 14:32", ["sensor", "security"]),
    ("Power consumption: 2.3kW, Solar output: 1.8kW", ["sensor", "energy"]),
    ("User entered code 1234 on keypad", ["sensor", "security"]),
    ("Device firmware installed successfully", ["system", "update"]),
    ("Network latency to gateway: 45ms", ["sensor", "network"]),
    ("User phone number registered", ["user"]),
    ("Payment wallet captured for billing", ["transaction"]),
    ("API credential stored in config", ["config"]),
]

for content, tags in events:
    record_id = brain.store(content, tags=tags)
    rec = brain.get(record_id)
    auto_tags = [t for t in rec.tags if t not in tags]
    extra = f"  -> protected tags: {auto_tags}" if auto_tags else ""
    print(f"  Stored: {content[:50]}...{extra}")

print("\n--- Local recall ---")
print(brain.recall("security events"))

print("\n--- Structured recall: energy data ---")
results = brain.recall_structured("power consumption solar", top_k=3)
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} - {r['content'][:60]}")

print("\n--- Circuit breaker: unreliable gateway ---")
brain.record_tool_failure("cloud_gateway")
brain.record_tool_failure("cloud_gateway")
brain.record_tool_failure("cloud_gateway")
if brain.is_tool_available("cloud_gateway"):
    print("  Gateway available")
else:
    print("  Gateway unavailable - staying local for now")
brain.record_tool_success("cloud_gateway")
print(f"  Gateway recovered: {brain.is_tool_available('cloud_gateway')}")

print("\n--- Maintenance ---")
config = MaintenanceConfig()
config.decay_enabled = True
config.consolidation_enabled = True
brain.configure_maintenance(config)
report = brain.run_maintenance()
print(f"  Records: {report.total_records}")
print(f"  Decayed: {report.decay.decayed}")
print(f"  Archived: {report.records_archived}")

print("\n--- Local benchmark ---")
start = time.perf_counter()
for i in range(100):
    brain.store(f"Sensor reading #{i}: value={i * 1.5:.1f}", tags=["benchmark"])
store_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
for _ in range(100):
    brain.recall("sensor reading", token_budget=500)
recall_ms = (time.perf_counter() - start) * 1000

print(f"  100 stores: {store_ms:.1f}ms")
print(f"  100 recalls: {recall_ms:.1f}ms")

stats = brain.stats()
print("\n--- Final stats ---")
print(f"  Total records: {stats['total_records']}")
print(f"  Encrypted: {brain.is_encrypted()}")

brain.close()
print("\nEdge brain saved to ./edge_data/ (encrypted)")
