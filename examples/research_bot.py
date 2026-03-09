"""Research bot - collect findings, track sources, synthesize results.

Demonstrates: research orchestrator, trust scoring, credibility, and insights.
"""

from aura import Aura, Level, TrustConfig

brain = Aura("./research_data")

tc = TrustConfig()
tc.source_trust = {
    "user": 1.0,
    "api": 0.9,
    "arxiv": 0.95,
    "wikipedia": 0.7,
    "web_scrape": 0.4,
    "reddit": 0.3,
}
brain.set_trust_config(tc)

print("=" * 60)
print("  Aura Research Bot - Trusted Knowledge Synthesis")
print("=" * 60)

project = brain.start_research("GRPO vs PPO for LLM training")
pid = project["id"]
print(f"\nProject: {project['topic']}")
print(f"Generated queries: {project.get('queries', [])}")

findings = [
    {
        "query": "What is GRPO?",
        "result": "GRPO is a group-based RL optimization approach used in modern LLM training.",
        "url": "https://arxiv.org/abs/2402.03300",
    },
    {
        "query": "GRPO vs PPO performance",
        "result": "Reports suggest GRPO can reduce training overhead while remaining competitive in quality.",
        "url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "query": "PPO limitations for LLM",
        "result": "PPO can require additional value-model overhead in some training setups.",
        "url": "https://arxiv.org/abs/1707.06347",
    },
    {
        "query": "Community feedback on GRPO",
        "result": "Community discussion suggests trade-offs between stability and scale.",
        "url": "https://reddit.com/r/LocalLLaMA/comments/example",
    },
]

for f in findings:
    brain.add_research_finding(pid, f["query"], f["result"], f["url"])
    print(f"\n  Added: {f['query']}")
    print(f"  Source credibility: {brain.get_credibility(f['url']):.2f}")

synthesis = (
    "GRPO appears to be a promising alternative to PPO for some LLM training workflows. "
    "It may reduce training overhead while remaining competitive in quality, but practical "
    "trade-offs still depend on scale, data, and engineering constraints."
)
synthesis_id = brain.complete_research(pid, synthesis=synthesis)
print(f"\n{'=' * 60}")
print(f"Research completed. Synthesis stored: {synthesis_id[:12]}")

print("\n--- Trust-weighted recall: 'GRPO memory usage' ---")
results = brain.recall_structured("GRPO memory usage", top_k=5)
for r in results:
    print(f"  score={r['score']:.3f} | {r['content'][:70]}")

print("\n--- Source credibility ---")
for src in [
    "https://arxiv.org/paper",
    "https://nature.com/article",
    "https://stackoverflow.com/q/123",
    "https://reddit.com/r/test",
    "https://random-blog.xyz/post",
]:
    print(f"  {src:45s} -> {brain.get_credibility(src):.2f}")

brain.set_credibility_override("internal-wiki.company.com", 0.95)
print(f"\n  Custom override: internal-wiki -> {brain.get_credibility('https://internal-wiki.company.com/doc'):.2f}")

print("\n--- Memory insights ---")
brain.store("GRPO reduces training cost in some setups", level=Level.Domain, tags=["grpo", "cost"])
brain.store("GRPO can reduce memory usage in some workflows", level=Level.Domain, tags=["grpo", "memory"])
brain.store("GRPO is associated with modern reasoning-model training", level=Level.Domain, tags=["grpo", "training"])

stats = brain.stats()
print(f"  Total records: {stats['total_records']}")
print(f"  Domain knowledge: {stats.get('domain', 0)}")

brain.close()
print("\nResearch data saved to ./research_data/")
