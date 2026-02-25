"""Run Aura maintenance as a background daemon.

This example shows how to use Aura's built-in background maintenance
thread, or alternatively run it manually in a loop.
"""

import signal
import sys
import time

from aura import Aura, MaintenanceConfig


def example_builtin_background():
    """Use Aura's built-in background thread."""
    brain = Aura("./daemon_data")

    # Configure maintenance
    config = MaintenanceConfig()
    config.decay_enabled = True
    config.consolidation_enabled = True
    brain.configure_maintenance(config)

    # Start background thread (runs every 60 seconds)
    brain.start_background(interval_secs=60)
    print("Background maintenance started")

    # Your agent does work here...
    brain.store("Example data being stored while maintenance runs")
    time.sleep(2)

    # Stop when done
    brain.stop_background()
    print("Background maintenance stopped")
    brain.close()


def example_manual_daemon():
    """Run maintenance manually in a loop with signal handling."""
    brain = Aura("./daemon_data")
    running = True

    def on_signal(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    interval = 120  # seconds
    print(f"Manual maintenance daemon started (interval={interval}s)")

    cycle = 0
    while running:
        cycle += 1
        try:
            report = brain.run_maintenance()
            print(
                f"[{cycle}] records={report.total_records} "
                f"decayed={report.decay.decayed} "
                f"merged={report.consolidation.native_merged} "
                f"archived={report.records_archived}"
            )
        except Exception as e:
            print(f"[{cycle}] ERROR: {e}", file=sys.stderr)

        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    brain.close()
    print("Daemon stopped.")


if __name__ == "__main__":
    if "--manual" in sys.argv:
        example_manual_daemon()
    else:
        example_builtin_background()
