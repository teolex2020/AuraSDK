import os

from aura import Aura


try:
    print("Starting Aura Memory Server with Dashboard UI...")

    brain = Aura("./test_brain")

    if brain.count() == 0:
        print("Brain is empty. Seeding sample memories for the dashboard.")
        brain.store("User asked for a lightweight dashboard.", tags=["ui", "request"])
        brain.store("Project uses local memory for agent context.", tags=["product"])
        brain.store("Team prefers concise operational summaries.", tags=["team", "preference"])
        brain.store("Follow up on deployment checklist this afternoon.", tags=["task"])

    brain.start_server(port=8000)

except AttributeError:
    print("Error: The python aura module was not compiled with the 'server' feature.")
    print("Please run: maturin develop --features server")
