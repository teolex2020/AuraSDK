"""Aura CLI — maintenance runner and interactive shell.

Usage:
    python -m aura maintain ./data --once          # Single cycle (cron)
    python -m aura maintain ./data --interval 120  # Daemon loop
    python -m aura status ./data                   # Last report
    python -m aura shell ./data                    # Interactive REPL
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from aura import Aura, MaintenanceConfig, TagTaxonomy, TrustConfig


def load_config(config_path: str) -> dict:
    """Load maintenance config from TOML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Minimal TOML parser (avoids external dep) — supports flat key=value
    # For full TOML, users should pip install tomli
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # pip install tomli
        except ImportError:
            print("TOML config requires Python 3.11+ or 'pip install tomli'", file=sys.stderr)
            sys.exit(1)

    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_config(brain: Aura, config: dict) -> None:
    """Apply parsed config dict to Aura instance."""
    if "taxonomy" in config:
        tax = TagTaxonomy()
        t = config["taxonomy"]
        if "identity_tags" in t:
            tax.identity_tags = set(t["identity_tags"])
        if "stable_tags" in t:
            tax.stable_tags = set(t["stable_tags"])
        if "volatile_tags" in t:
            tax.volatile_tags = set(t["volatile_tags"])
        brain.set_taxonomy(tax)

    if "trust" in config:
        tc = TrustConfig()
        if "source_trust" in config["trust"]:
            tc.source_trust = config["trust"]["source_trust"]
        brain.set_trust_config(tc)

    if "maintenance" in config:
        mc = MaintenanceConfig()
        m = config["maintenance"]
        if "decay_enabled" in m:
            mc.decay_enabled = m["decay_enabled"]
        if "consolidation_enabled" in m:
            mc.consolidation_enabled = m["consolidation_enabled"]
        if "max_clusters_per_run" in m:
            mc.max_clusters_per_run = m["max_clusters_per_run"]
        brain.configure_maintenance(mc)


def cmd_maintain(args: argparse.Namespace) -> None:
    """Run maintenance: single cycle or continuous loop."""
    brain = Aura(args.path)

    if args.config:
        config = load_config(args.config)
        apply_config(brain, config)

    if args.once:
        report = brain.run_maintenance()
        print(f"Maintenance complete at {report.timestamp}")
        print(f"  Records: {report.total_records}")
        print(f"  Decayed: {report.decay.decayed}, Archived (decay): {report.decay.archived}")
        print(f"  Promoted: {report.reflect.promoted}, Archived (reflect): {report.reflect.archived}")
        print(f"  Insights: {report.insights_found}")
        print(f"  Consolidated: {report.consolidation.native_merged}")
        print(f"  Cross-connections: {report.cross_connections}")
        print(f"  Records archived: {report.records_archived}")
        if report.task_reminders:
            print(f"  Task reminders:")
            for r in report.task_reminders:
                print(f"    - {r}")
        brain.close()
        return

    # Daemon mode
    interval = args.interval or 120
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False
        print("\nStopping maintenance daemon...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Starting maintenance daemon (interval={interval}s, path={args.path})")
    cycle = 0
    while running:
        cycle += 1
        try:
            report = brain.run_maintenance()
            print(
                f"[cycle {cycle}] records={report.total_records} "
                f"decayed={report.decay.decayed} archived={report.records_archived} "
                f"promoted={report.reflect.promoted} merged={report.consolidation.native_merged}"
            )
        except Exception as e:
            print(f"[cycle {cycle}] ERROR: {e}", file=sys.stderr)

        # Sleep in small increments to respond to signals
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    brain.close()
    print("Daemon stopped.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show brain status and stats."""
    brain = Aura(args.path)
    stats = brain.stats()
    print(f"Aura Brain: {args.path}")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Working:   {stats.get('working', 0)}")
    print(f"  Decisions: {stats.get('decisions', 0)}")
    print(f"  Domain:    {stats.get('domain', 0)}")
    print(f"  Identity:  {stats.get('identity', 0)}")
    print(f"  Connections: {stats.get('total_connections', 0)}")
    print(f"  Tags: {stats.get('total_tags', 0)}")
    brain.close()


def cmd_mcp(args: argparse.Namespace) -> None:
    """Run Aura as an MCP server over stdio."""
    from aura.mcp_server import run_mcp
    run_mcp(path=args.path, password=args.password)


def cmd_shell(args: argparse.Namespace) -> None:
    """Interactive REPL for store/recall/search."""
    brain = Aura(args.path)
    print(f"Aura Shell — {args.path}")
    print("Commands: store <text>, recall <query>, search <query>, stats, quit")
    print()

    while True:
        try:
            line = input("aura> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "store":
            if not arg:
                print("Usage: store <text>")
                continue
            record_id = brain.store(arg)
            print(f"Stored: {record_id}")
        elif cmd == "recall":
            if not arg:
                print("Usage: recall <query>")
                continue
            result = brain.recall(arg)
            print(result or "(no results)")
        elif cmd == "search":
            if not arg:
                print("Usage: search <query>")
                continue
            results = brain.search(arg)
            if not results:
                print("(no results)")
            for r in results:
                print(f"  [{r.level}] {r.id}: {r.content[:80]}")
        elif cmd == "stats":
            stats = brain.stats()
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v}")
        elif cmd == "delete":
            if not arg:
                print("Usage: delete <record_id>")
                continue
            if brain.delete(arg):
                print(f"Deleted: {arg}")
            else:
                print(f"Not found: {arg}")
        elif cmd == "get":
            if not arg:
                print("Usage: get <record_id>")
                continue
            rec = brain.get(arg)
            if rec:
                print(f"  ID: {rec.id}")
                print(f"  Content: {rec.content}")
                print(f"  Level: {rec.level}")
                print(f"  Strength: {rec.strength:.3f}")
                print(f"  Tags: {rec.tags}")
            else:
                print(f"Not found: {arg}")
        elif cmd == "maintain":
            report = brain.run_maintenance()
            print(f"Maintenance done: decayed={report.decay.decayed} archived={report.records_archived}")
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: store, recall, search, stats, get, delete, maintain, quit")

    brain.close()
    print("Bye!")


def main():
    parser = argparse.ArgumentParser(
        prog="aura",
        description="Aura — Cognitive Memory for AI Agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # maintain
    p_maintain = subparsers.add_parser("maintain", help="Run maintenance cycle(s)")
    p_maintain.add_argument("path", help="Path to brain data directory")
    p_maintain.add_argument("--once", action="store_true", help="Single cycle then exit")
    p_maintain.add_argument("--interval", type=int, help="Seconds between cycles (daemon mode)")
    p_maintain.add_argument("--config", help="Path to maintenance.toml config")

    # status
    p_status = subparsers.add_parser("status", help="Show brain status")
    p_status.add_argument("path", help="Path to brain data directory")

    # shell
    p_shell = subparsers.add_parser("shell", help="Interactive REPL")
    p_shell.add_argument("path", help="Path to brain data directory")

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Run MCP server (stdio)")
    p_mcp.add_argument("path", nargs="?", default="./aura_brain",
                        help="Path to brain data directory (default: ./aura_brain)")
    p_mcp.add_argument("--password", help="Encryption password")

    # serve
    p_serve = subparsers.add_parser("serve", help="Run MCP HTTP+SSE server (Make.com, n8n, remote)")
    p_serve.add_argument("path", nargs="?", default="./aura_brain",
                          help="Path to brain data directory (default: ./aura_brain)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p_serve.add_argument("--password", help="Encryption password")

    args = parser.parse_args()

    if args.command == "maintain":
        cmd_maintain(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "shell":
        cmd_shell(args)
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "serve":
        from aura.mcp_http import run_http
        run_http(path=args.path, host=args.host, port=args.port, password=args.password)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
