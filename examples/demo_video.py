"""Aura SDK -- Scripted Terminal Demo for Video Recording.

Auto-running demo that showcases every major Aura feature in ~60 seconds.
No user input needed. Optimized for screen recording.

Run:
    python examples/demo_video.py
"""

import sys
import os
import time
import shutil

# Windows ANSI support + encoding fix
if sys.platform == "win32":
    os.system("")  # enable VT100 processing on Windows
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aura import Aura, Level, TrustConfig, AgentPersona

# -- ANSI Colors --
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"

DEMO_DIR = "./demo_video_data"
DEMO_ENC = "./demo_video_encrypted"

section_num = 0


def header(title):
    global section_num
    section_num += 1
    line = "-" * 60
    print(f"\n{CYAN}{line}{RESET}")
    print(f"{CYAN}  [{section_num}/8] {title}{RESET}")
    print(f"{CYAN}{line}{RESET}\n")


def ms(t0):
    return (time.perf_counter() - t0) * 1000


def kv(key, value, indent=2):
    pad = " " * indent
    print(f"{pad}{YELLOW}{key}:{RESET} {GREEN}{value}{RESET}")


def pause(s=0.8):
    time.sleep(s)


def banner(text):
    line = "=" * 60
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    for t in text:
        print(f"{BOLD}{CYAN}  {t}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}\n")


def main():
    total_t0 = time.perf_counter()

    banner([
        "Aura SDK -- Meet Your Agent's Brain",
        "Cognitive memory for AI agents",
        "No LLM | No cloud | <1ms recall",
    ])
    pause(1.0)

    # Clean up from previous runs
    for d in [DEMO_DIR, DEMO_ENC]:
        shutil.rmtree(d, ignore_errors=True)

    brain = Aura(DEMO_DIR)

    # Warm-up (cold cache mitigation)
    brain.store("warmup", tags=["warmup"])
    brain.recall("warmup")
    brain.delete(brain.search(tags=["warmup"])[0].id)

    # ================================================================
    # SECTION 1: Agent Identity
    # ================================================================
    header("AGENT IDENTITY")

    persona = AgentPersona()
    persona.name = "Atlas"
    persona.role = "Research Assistant"
    brain.set_persona(persona)
    kv("Agent", "Atlas (Research Assistant)")

    brain.store_user_profile({
        "name": "Teo",
        "role": "Developer",
        "language": "Ukrainian",
    })
    kv("User", "Teo (Developer)")
    kv("Records", brain.count())
    pause()

    # ================================================================
    # SECTION 2: 4-Level Memory Hierarchy
    # ================================================================
    header("4-LEVEL MEMORY HIERARCHY")

    memories = [
        ("User prefers concise answers and dark mode", Level.Identity, ["preference", "ui"]),
        ("Rust ownership prevents data races at compile time", Level.Domain, ["rust", "lang"]),
        ("We chose PostgreSQL over MongoDB for ACID compliance", Level.Decisions, ["database", "architecture"]),
        ("Fix auth bug: users getting 403 on /api/auth endpoint", Level.Working, ["bug", "auth"]),
    ]

    level_info = {
        "IDENTITY": ("core", "0.99"),
        "DOMAIN": ("core", "0.95"),
        "DECISIONS": ("cognitive", "0.90"),
        "WORKING": ("cognitive", "0.80"),
    }

    ids = []
    for content, level, tags in memories:
        t0 = time.perf_counter()
        rid = brain.store(content, level=level, tags=tags)
        elapsed = ms(t0)
        ids.append(rid)
        lname = str(level).split(".")[-1]
        tier, decay = level_info[lname]
        print(f"  {YELLOW}{lname:10}{RESET} | {DIM}{tier:9}{RESET} | {DIM}decay={decay}{RESET} | {MAGENTA}{elapsed:.2f}ms{RESET}")
        print(f"  {DIM}  {content[:55]}...{RESET}")
        time.sleep(0.2)

    print(f"\n  {GREEN}Total records: {brain.count()}{RESET}")
    pause()

    # ================================================================
    # SECTION 3: RRF Fusion Recall
    # ================================================================
    header("RRF FUSION RECALL (<1ms)")

    # Formatted recall
    t0 = time.perf_counter()
    context = brain.recall("authentication issues", token_budget=2000)
    elapsed = ms(t0)

    print(f"  {MAGENTA}recall() latency: {elapsed:.2f}ms{RESET}")
    print(f"  {DIM}No LLM. No embeddings. Pure Rust computation.{RESET}\n")
    for line in context.strip().split("\n"):
        if line.strip():
            print(f"  {DIM}{line}{RESET}")

    pause(0.5)

    # Structured recall
    t0 = time.perf_counter()
    results = brain.recall_structured("rust performance", top_k=5)
    elapsed = ms(t0)

    print(f"\n  {MAGENTA}recall_structured() latency: {elapsed:.2f}ms{RESET}\n")
    for r in results:
        score = r["score"]
        lvl = r["level"]
        txt = r["content"][:50]
        print(f"  {YELLOW}[{lvl:10}]{RESET} score={GREEN}{score:.3f}{RESET} | {txt}")
    pause()

    # ================================================================
    # SECTION 4: Trust & Provenance
    # ================================================================
    header("TRUST & PROVENANCE")

    tc = TrustConfig()
    tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
    brain.set_trust_config(tc)

    brain.store("Python 3.13 released with JIT compiler", level=Level.Domain,
                tags=["python"], channel="user")
    brain.store("Python 3.13 might have JIT compiler", level=Level.Working,
                tags=["python"], channel="web_scrape")

    results = brain.recall_structured("Python JIT", top_k=2)
    print(f"  {BOLD}Same topic, different sources:{RESET}\n")
    for r in results:
        trust = r.get("trust", "?")
        src = r.get("source", "?")
        bar_len = int(float(r["score"]) * 20)
        bar = "=" * bar_len + " " * (20 - bar_len)
        print(f"  [{bar}] {GREEN}{r['score']:.3f}{RESET}  trust={trust}  src={src}")
        print(f"    {DIM}{r['content'][:60]}{RESET}")

    print(f"\n  {BOLD}Domain credibility (60+ pre-scored):{RESET}")
    for domain in ["arxiv.org", "nature.com", "reddit.com", "unknown-blog.xyz"]:
        cred = brain.get_credibility(domain)
        bar_len = int(cred * 20)
        bar = "=" * bar_len + " " * (20 - bar_len)
        print(f"  [{bar}] {cred:.2f}  {domain}")
    pause()

    # ================================================================
    # SECTION 5: Knowledge Graph
    # ================================================================
    header("KNOWLEDGE GRAPH")

    # Connect auth bug with database decision
    id_bug = ids[3]   # Working: auth bug
    id_db = ids[2]    # Decisions: PostgreSQL
    brain.connect(id_bug, id_db, weight=0.9, relationship="causal")

    stats = brain.stats()
    kv("Connections", stats.get("total_connections", 0))
    kv("Tags tracked", stats.get("total_tags", 0))

    t0 = time.perf_counter()
    results = brain.recall_structured("database auth", top_k=3, expand_connections=True)
    elapsed = ms(t0)
    print(f"\n  {MAGENTA}Recall with connection expansion: {elapsed:.2f}ms{RESET}")
    for r in results:
        print(f"  {YELLOW}[{r['level']:10}]{RESET} {r['content'][:55]}")
    pause()

    # ================================================================
    # SECTION 6: Maintenance Cycle
    # ================================================================
    header("8-PHASE MAINTENANCE CYCLE")

    t0 = time.perf_counter()
    report = brain.run_maintenance()
    elapsed = ms(t0)

    print(f"  {MAGENTA}Full cycle: {elapsed:.2f}ms{RESET}\n")
    print(f"  Decayed:      {YELLOW}{report.decay.decayed}{RESET}     Promoted:   {YELLOW}{report.reflect.promoted}{RESET}")
    print(f"  Merged:       {YELLOW}{report.consolidation.native_merged}{RESET}     Archived:   {YELLOW}{report.records_archived}{RESET}")
    print(f"  Insights:     {YELLOW}{report.insights_found}{RESET}     Cross-conn: {YELLOW}{report.cross_connections}{RESET}")
    print(f"\n  {GREEN}Total records after cycle: {report.total_records}{RESET}")
    pause()

    # ================================================================
    # SECTION 7: Two-Tier Architecture
    # ================================================================
    header("TWO-TIER ARCHITECTURE")

    ts = brain.tier_stats()
    print(f"  {BOLD}Cognitive Tier{RESET} {DIM}(ephemeral, fast decay){RESET}")
    print(f"    +-- Working:   {YELLOW}{ts.get('cognitive_working', 0)}{RESET}")
    print(f"    +-- Decisions: {YELLOW}{ts.get('cognitive_decisions', 0)}{RESET}")
    print()
    print(f"  {BOLD}Core Tier{RESET} {DIM}(permanent, slow decay){RESET}")
    print(f"    +-- Domain:    {YELLOW}{ts.get('core_domain', 0)}{RESET}")
    print(f"    +-- Identity:  {YELLOW}{ts.get('core_identity', 0)}{RESET}")

    cog = brain.recall_cognitive("bug")
    core = brain.recall_core_tier("user preferences")
    print(f"\n  recall_cognitive('bug'):           {GREEN}{len(cog)} results{RESET}")
    print(f"  recall_core_tier('preferences'):   {GREEN}{len(core)} results{RESET}")
    pause()

    # ================================================================
    # SECTION 8: Encryption
    # ================================================================
    header("ENCRYPTION AT REST")

    brain2 = Aura(DEMO_ENC, password="demo-secret")
    brain2.store("Top secret: deployment key is XK-42-ZZ", tags=["secret"])
    kv("Encrypted", brain2.is_encrypted())
    brain2.close()

    # Reopen and verify
    brain2 = Aura(DEMO_ENC, password="demo-secret")
    result = brain2.recall("deployment key")
    print(f"\n  {DIM}Closed, reopened with password -- data persists:{RESET}")
    for line in result.strip().split("\n"):
        if line.strip():
            print(f"  {DIM}{line}{RESET}")
    kv("Algorithm", "ChaCha20-Poly1305 + Argon2id")
    brain2.close()
    pause()

    # ================================================================
    # FINALE
    # ================================================================
    brain.close()
    total_elapsed = ms(total_t0)

    banner([
        "Aura SDK -- Cognitive Memory for AI Agents",
        "",
        f"  Total demo time:  {total_elapsed / 1000:.1f}s",
        f"  Records created:  {report.total_records}",
        f"  Recall latency:   <1ms",
        "",
        "  pip install aura-memory",
        "  github.com/teolex2020/AuraSDK",
    ])

    # Cleanup
    for d in [DEMO_DIR, DEMO_ENC]:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{RED}Interrupted.{RESET}")
    finally:
        for d in [DEMO_DIR, DEMO_ENC]:
            shutil.rmtree(d, ignore_errors=True)
