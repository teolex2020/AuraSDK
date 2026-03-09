"""Using Aura as persistent memory for an AI agent.

Demonstrates: trust, provenance, research, circuit breaker, and maintenance.
"""

from aura import AgentPersona, Aura, Level, MaintenanceConfig, TagTaxonomy, TrustConfig


def main():
    brain = Aura("./agent_data")

    persona = AgentPersona()
    persona.name = "Atlas"
    persona.role = "Research Assistant"
    brain.set_persona(persona)

    brain.store_user_profile(
        {
            "name": "Teo",
            "language": "Ukrainian",
            "role": "Developer",
            "preference": "concise answers",
        }
    )

    tc = TrustConfig()
    tc.source_trust = {"user": 1.0, "api": 0.8, "agent": 0.7, "web_scrape": 0.5}
    brain.set_trust_config(tc)

    tax = TagTaxonomy()
    tax.identity_tags = {"name", "role", "language", "preference"}
    tax.volatile_tags = {"task", "temp", "draft"}
    tax.sensitive_tags = {"credential", "financial", "contact"}
    brain.set_taxonomy(tax)

    brain.store(
        "I prefer TypeScript over JavaScript for large projects",
        level=Level.Decisions,
        tags=["preference", "typescript"],
        channel="user",
    )
    brain.store(
        "TypeScript 5.4 adds NoInfer utility type",
        level=Level.Domain,
        tags=["typescript", "release"],
        channel="api",
    )
    brain.store(
        "Some blog says TypeScript is slow",
        level=Level.Working,
        tags=["typescript", "performance"],
        channel="web_scrape",
    )

    results = brain.recall_structured("typescript", top_k=5)
    print("--- Recall with trust-aware ranking ---")
    for r in results:
        print(f"  score={r['score']:.3f} trust={r.get('trust')} src={r.get('source')} - {r['content'][:50]}")

    project = brain.start_research("Rust async patterns")
    pid = project["id"]
    brain.add_research_finding(pid, "What is Tokio?", "Tokio is a popular async runtime for Rust", "https://tokio.rs")
    brain.add_research_finding(pid, "Async traits", "Rust stabilized async fn in traits")
    brain.add_research_finding(pid, "Structured concurrency", "JoinSet helps manage concurrent tasks")
    synthesis_id = brain.complete_research(pid)
    print("\n--- Research completed ---")
    print(f"  Synthesis record: {synthesis_id[:12]}")

    brain.record_tool_failure("weather_api")
    brain.record_tool_failure("weather_api")
    brain.record_tool_failure("weather_api")
    if brain.is_tool_available("weather_api"):
        print("\nWeather API is available")
    else:
        health = brain.tool_health()
        print(f"\nWeather API circuit: {health.get('weather_api', 'unknown')}")
        print("  Using cached data instead...")
    brain.record_tool_success("weather_api")

    print("\n--- Source credibility ---")
    for src in ["nature.com", "arxiv.org", "reddit.com", "medium.com", "unknown-blog.xyz"]:
        print(f"  {src}: {brain.get_credibility(src):.2f}")

    config = MaintenanceConfig()
    config.decay_enabled = True
    config.consolidation_enabled = True
    brain.configure_maintenance(config)
    report = brain.run_maintenance()
    print("\n--- Maintenance report ---")
    print(f"  Total records: {report.total_records}")
    print(f"  Decayed: {report.decay.decayed}")
    print(f"  Promoted: {report.reflect.promoted}")
    print(f"  Merged: {report.consolidation.native_merged}")
    print(f"  Archived: {report.records_archived}")

    profile = brain.get_user_profile()
    persona_obj = brain.get_persona()
    print("\n--- Identity ---")
    print(f"  User: {profile.get('name', 'N/A') if profile else 'N/A'}")
    print(f"  Agent: {persona_obj.name if persona_obj else 'N/A'}")

    brain.close()


if __name__ == "__main__":
    main()
