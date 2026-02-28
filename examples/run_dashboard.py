import os
from aura import Aura

try:
    print("Starting Aura Memory Server with Dashboard UI...")
    
    # Initialize the brain
    brain = Aura("./test_brain")
    
    # Ensure some data exists
    if brain.count() == 0:
        print("Brain is empty. Seeding some initial memories for the dashboard to display.")
        brain.store("User explicitly requested a lightweight Vanilla JS dashboard for Aura.", tags=["ui", "preferences"])
        brain.store("AuraSDK uses a pure Rust implementation with SDR + MinHash for retrieval.", tags=["architecture"])
        brain.store("The background maintenance cycle consists of 8 phases.", level="domain", tags=["core"])
        brain.store("Memory decays based on retention level: Working (0.8), Decisions (0.9)....")

    # Start the HTTP server (available if compiled with server feature)
    # The pure Rust backend handles serving /ui/index.html
    brain.start_server(port=8000)

except AttributeError:
    print("Error: The python aura module was not compiled with the 'server' feature.")
    print("Please run: maturin develop --features server")
