"""Research bot — collect findings, track sources, synthesize results.

Demonstrates: research orchestrator, trust scoring, credibility, insights.

Run:
    python examples/research_bot.py
"""

from aura import Aura, Level, TrustConfig

brain = Aura("./research_data")

# ── Configure trust scoring ──
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

# ── Start a research project ──
print("=" * 60)
print("  Aura Research Bot — Trusted Knowledge Synthesis")
print("=" * 60)

project = brain.start_research("GRPO vs PPO for LLM training")
pid = project["id"]
print(f"\nProject: {project['topic']}")
print(f"Generated queries: {project.get('queries', [])}")

# ── Simulate collecting findings from various sources ──
findings = [
    {
        "query": "What is GRPO?",
        "result": "Group Relative Policy Optimization (GRPO) eliminates the critic model "
                  "by using group-based advantage estimation. Published by DeepSeek, 2024. "
                  "Reduces memory by ~50% compared to PPO.",
        "url": "https://arxiv.org/abs/2402.03300",
    },
    {
        "query": "GRPO vs PPO performance",
        "result": "In DeepSeek-R1 experiments, GRPO achieved comparable reasoning performance "
                  "to PPO while using significantly less GPU memory and training time.",
        "url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "query": "PPO limitations for LLM",
        "result": "PPO requires a separate critic/value model of similar size to the policy "
                  "model, roughly doubling GPU memory requirements during training.",
        "url": "https://arxiv.org/abs/1707.06347",
    },
    {
        "query": "Community feedback on GRPO",
        "result": "Some Reddit users report GRPO is unstable on small datasets, "
                  "but great for large-scale training. Take with a grain of salt.",
        "url": "https://reddit.com/r/LocalLLaMA/comments/example",
    },
]

for f in findings:
    brain.add_research_finding(pid, f["query"], f["result"], f["url"])
    credibility = brain.get_credibility(f["url"])
    print(f"\n  Added: {f['query']}")
    print(f"  Source credibility: {credibility:.2f}")

# ── Complete research with synthesis ──
synthesis = (
    "GRPO (Group Relative Policy Optimization) is a compelling alternative to PPO "
    "for LLM RLHF training. Key advantage: eliminates the critic model, reducing "
    "GPU memory by ~50%. DeepSeek's experiments show comparable performance to PPO. "
    "Trade-off: may be less stable on small datasets. Recommended for teams with "
    "large-scale training infrastructure."
)
synthesis_id = brain.complete_research(pid, synthesis=synthesis)
print(f"\n{'=' * 60}")
print(f"Research completed. Synthesis stored: {synthesis_id[:12]}")

# ── Recall with trust weighting ──
print(f"\n--- Trust-weighted recall: 'GRPO memory usage' ---")
results = brain.recall_structured("GRPO memory usage", top_k=5)
for r in results:
    src = r.get("source", "unknown")
    print(f"  score={r['score']:.3f} | {r['content'][:70]}")

# ── Source credibility comparison ──
print(f"\n--- Source credibility ---")
sources = [
    "https://arxiv.org/paper",
    "https://nature.com/article",
    "https://stackoverflow.com/q/123",
    "https://reddit.com/r/test",
    "https://random-blog.xyz/post",
]
for src in sources:
    print(f"  {src:45s} -> {brain.get_credibility(src):.2f}")

# Override credibility for your internal sources
brain.set_credibility_override("internal-wiki.company.com", 0.95)
print(f"\n  Custom override: internal-wiki -> {brain.get_credibility('https://internal-wiki.company.com/doc'):.2f}")

# ── Run insights ──
print(f"\n--- Memory insights ---")
brain.store("GRPO reduces training cost by 50%", level=Level.Domain, tags=["grpo", "cost"])
brain.store("GRPO reduces memory usage by 50%", level=Level.Domain, tags=["grpo", "memory"])
brain.store("GRPO was published by DeepSeek", level=Level.Domain, tags=["grpo", "deepseek"])

stats = brain.stats()
print(f"  Total records: {stats['total_records']}")
print(f"  Domain knowledge: {stats.get('domain', 0)}")

brain.close()
print("\nResearch data saved to ./research_data/")
