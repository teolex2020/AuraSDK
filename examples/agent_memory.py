"""Using Aura as persistent memory for an AI agent.

Demonstrates: trust, provenance, research, circuit breaker, maintenance.
"""

from aura import (
    Aura,
    Level,
    TrustConfig,
    TagTaxonomy,
    MaintenanceConfig,
    AgentPersona,
)


def main():
    brain = Aura("./agent_data")

    # ── 1. Configure the agent's identity ──
    persona = AgentPersona()
    persona.name = "Atlas"
    persona.role = "Research Assistant"
    brain.set_persona(persona)

    brain.store_user_profile({
        "name": "Teo",
        "language": "Ukrainian",
        "role": "Developer",
        "preference": "concise answers",
    })

    # ── 2. Configure trust scoring ──
    tc = TrustConfig()
    tc.source_trust = {
        "user": 1.0,      # Direct user input is fully trusted
        "api": 0.8,       # API responses are mostly trusted
        "agent": 0.7,     # Agent-generated content
        "web_scrape": 0.5, # Web scraping is less trusted
    }
    brain.set_trust_config(tc)

    # ── 3. Configure tag taxonomy ──
    tax = TagTaxonomy()
    tax.identity_tags = {"name", "role", "language", "preference"}
    tax.volatile_tags = {"task", "temp", "draft"}
    tax.sensitive_tags = {"credential", "financial", "contact"}
    brain.set_taxonomy(tax)

    # ── 4. Store memories from different sources ──
    # User says something directly
    brain.store(
        "I prefer TypeScript over JavaScript for large projects",
        level=Level.Decisions,
        tags=["preference", "typescript"],
        channel="user",
    )

    # Agent discovers something via API
    brain.store(
        "TypeScript 5.4 adds NoInfer utility type",
        level=Level.Domain,
        tags=["typescript", "release"],
        channel="api",
    )

    # Agent scrapes a website
    brain.store(
        "Some blog says TypeScript is slow",
        level=Level.Working,
        tags=["typescript", "performance"],
        channel="web_scrape",
    )

    # ── 5. Recall with trust weighting ──
    # High-trust sources rank higher in results
    results = brain.recall_structured("typescript", top_k=5)
    print("--- Recall with trust weighting ---")
    for r in results:
        print(f"  score={r['score']:.3f} trust={r['trust']} src={r['source']} — {r['content'][:50]}")

    # ── 6. Research orchestrator ──
    project = brain.start_research("Rust async patterns")
    pid = project["id"]

    brain.add_research_finding(pid,
        "What is Tokio?",
        "Tokio is the most popular async runtime for Rust",
        "https://tokio.rs")
    brain.add_research_finding(pid,
        "Async traits",
        "Rust 1.75 stabilized async fn in traits")
    brain.add_research_finding(pid,
        "Structured concurrency",
        "Use JoinSet for structured concurrency in Tokio")

    synthesis_id = brain.complete_research(pid)
    print(f"\n--- Research completed ---")
    print(f"  Synthesis record: {synthesis_id[:12]}")

    # ── 7. Circuit breaker ──
    # Simulate an unreliable tool
    brain.record_tool_failure("weather_api")
    brain.record_tool_failure("weather_api")
    brain.record_tool_failure("weather_api")

    if brain.is_tool_available("weather_api"):
        print("\nWeather API is available")
    else:
        print(f"\nWeather API circuit is {brain.tool_health('weather_api')}")
        print("  Using cached data instead...")

    # After the API recovers
    brain.record_tool_success("weather_api")

    # ── 8. Credibility check ──
    sources = ["nature.com", "arxiv.org", "reddit.com", "medium.com", "unknown-blog.xyz"]
    print("\n--- Source credibility ---")
    for src in sources:
        print(f"  {src}: {brain.get_credibility(src):.2f}")

    # ── 9. Run maintenance ──
    config = MaintenanceConfig()
    config.decay_enabled = True
    config.consolidation_enabled = True
    brain.configure_maintenance(config)

    report = brain.run_maintenance()
    print(f"\n--- Maintenance report ---")
    print(f"  Total records: {report.total_records}")
    print(f"  Decayed: {report.decay.decayed}")
    print(f"  Promoted: {report.reflect.promoted}")
    print(f"  Merged: {report.consolidation.native_merged}")
    print(f"  Archived: {report.records_archived}")

    # ── 10. Check user profile and persona ──
    profile = brain.get_user_profile()
    persona = brain.get_persona()
    print(f"\n--- Identity ---")
    print(f"  User: {profile}")
    print(f"  Agent: {persona}")

    brain.close()


if __name__ == "__main__":
    main()
