"""Aura SDK -- scripted product demo.

This example is intentionally public-facing: it shows user-visible behavior
without explaining internal implementation details.

Run:
    python examples/demo_video.py
"""

import os
import shutil
import sys
import time

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aura import AgentPersona, Aura, Level, TrustConfig

BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"

DEMO_DIR = "./demo_video_data"
DEMO_ENC = "./demo_video_encrypted"
section_num = 0


def header(title: str) -> None:
    global section_num
    section_num += 1
    line = "-" * 60
    print(f"\n{CYAN}{line}{RESET}")
    print(f"{CYAN}  [{section_num}/8] {title}{RESET}")
    print(f"{CYAN}{line}{RESET}\n")


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def pause(seconds: float = 0.6) -> None:
    time.sleep(seconds)


def banner(lines: list[str]) -> None:
    line = "=" * 60
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    for text in lines:
        print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}\n")


def main() -> None:
    for path in [DEMO_DIR, DEMO_ENC]:
        shutil.rmtree(path, ignore_errors=True)

    total_t0 = time.perf_counter()
    banner(
        [
            "Aura SDK -- Local Memory for AI Agents",
            "Persistent context without external services",
            "Fast recall, trust-aware storage, encrypted local data",
        ]
    )

    brain = Aura(DEMO_DIR)

    header("Agent Identity")
    persona = AgentPersona()
    persona.name = "Atlas"
    persona.role = "Research Assistant"
    brain.set_persona(persona)
    brain.store_user_profile({"name": "Teo", "role": "Developer", "language": "Ukrainian"})
    print(f"  {YELLOW}Agent:{RESET} {GREEN}Atlas{RESET}")
    print(f"  {YELLOW}User:{RESET} {GREEN}Teo{RESET}")
    pause()

    header("Store Memories")
    samples = [
        ("User prefers concise answers and dark mode", Level.Identity, ["preference", "ui"]),
        ("Rust ownership helps prevent common bugs", Level.Domain, ["rust", "lang"]),
        ("Use PostgreSQL for billing data", Level.Decisions, ["database", "work"]),
        ("Fix blocked-user auth issue", Level.Working, ["bug", "auth"]),
    ]
    for content, level, tags in samples:
        t0 = time.perf_counter()
        brain.store(content, level=level, tags=tags)
        print(
            f"  {YELLOW}{str(level).split('.')[-1]:10}{RESET} | "
            f"{MAGENTA}{elapsed_ms(t0):.2f}ms{RESET} | {DIM}{content[:50]}{RESET}"
        )
    pause()

    header("Recall Context")
    t0 = time.perf_counter()
    context = brain.recall("authentication issues", token_budget=1200)
    print(f"  {MAGENTA}recall(): {elapsed_ms(t0):.2f}ms{RESET}\n")
    for line in context.strip().splitlines():
        if line.strip():
            print(f"  {DIM}{line}{RESET}")
    pause()

    header("Trust-Aware Results")
    tc = TrustConfig()
    tc.source_trust = {"user": 1.0, "api": 0.8, "web_scrape": 0.5}
    brain.set_trust_config(tc)
    brain.store("Python release notes mention performance improvements", level=Level.Domain, tags=["python"], channel="user")
    brain.store("A forum post claims major Python speedups", level=Level.Working, tags=["python"], channel="web_scrape")
    results = brain.recall_structured("python performance", top_k=2)
    for r in results:
        print(f"  score={GREEN}{r['score']:.3f}{RESET} trust={r.get('trust', '?')} src={r.get('source', '?')}")
        print(f"    {DIM}{r['content'][:60]}{RESET}")
    pause()

    header("Connected Knowledge")
    auth_records = brain.search(tags=["auth"])
    decision_records = brain.search(level=Level.Decisions)
    if auth_records and decision_records:
        brain.connect(auth_records[0].id, decision_records[0].id, weight=0.9, relationship="causal")
    results = brain.recall_structured("database auth", top_k=3, expand_connections=True)
    for r in results:
        print(f"  {YELLOW}[{r['level']:10}]{RESET} {r['content'][:55]}")
    pause()

    header("Maintenance")
    t0 = time.perf_counter()
    report = brain.run_maintenance()
    print(f"  {MAGENTA}maintenance(): {elapsed_ms(t0):.2f}ms{RESET}")
    print(f"  Decayed: {report.decay.decayed} | Promoted: {report.reflect.promoted}")
    print(f"  Merged: {report.consolidation.native_merged} | Archived: {report.records_archived}")
    print(f"  Insights: {report.insights_found}")
    pause()

    header("Tiered Memory")
    tier = brain.tier_stats()
    print(f"  Cognitive tier: {tier.get('cognitive_total', 0)}")
    print(f"  Core tier:      {tier.get('core_total', 0)}")
    print(f"  Cognitive recall results: {len(brain.recall_cognitive('bug'))}")
    print(f"  Core recall results:      {len(brain.recall_core_tier('preferences'))}")
    pause()

    header("Encrypted Storage")
    secure = Aura(DEMO_ENC, password="demo-secret")
    secure.store("Private deployment note", tags=["private"])
    secure.close()
    secure = Aura(DEMO_ENC, password="demo-secret")
    print(f"  {YELLOW}Encrypted:{RESET} {GREEN}{secure.is_encrypted()}{RESET}")
    print(f"  {YELLOW}Recovered:{RESET} {GREEN}{bool(secure.recall('deployment note').strip())}{RESET}")
    secure.close()

    brain.close()
    total = elapsed_ms(total_t0) / 1000
    banner(
        [
            "Aura SDK -- Local Agent Memory",
            "",
            f"Total demo time: {total:.1f}s",
            f"Records retained: {report.total_records}",
            "",
            "pip install aura-memory",
            "github.com/teolex2020/AuraSDK",
        ]
    )

    for path in [DEMO_DIR, DEMO_ENC]:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        for path in [DEMO_DIR, DEMO_ENC]:
            shutil.rmtree(path, ignore_errors=True)
