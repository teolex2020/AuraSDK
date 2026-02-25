"""Encrypted memory brain.

All data is encrypted at rest with ChaCha20-Poly1305.
Key derivation uses Argon2id.
"""

from aura import Aura, Level


def main():
    # In production, use: password = os.environ["AURA_PASSWORD"]
    password = "demo-password-change-me"

    # Create encrypted brain
    brain = Aura("./encrypted_data", password=password)
    print(f"Encrypted: {brain.is_encrypted()}")

    # Store sensitive data — auto-protect guards will detect and tag PII
    brain.store("Server config: region=us-east-1", tags=["config"])
    brain.store("Contact support at help@example.com", tags=["contact"])
    brain.store("Meeting notes from Q4 planning session", tags=["notes"])

    # Data is encrypted on disk — file contents are unreadable without password
    brain.flush()

    # Recall works normally
    results = brain.recall_structured("credentials", top_k=5)
    for r in results:
        print(f"  [{r['level']}] {r['content'][:40]}...")

    # Export (decrypted JSON)
    export = brain.export_json()
    print(f"\nExported {len(export)} chars of JSON")

    brain.close()

    # Reopen with same password
    brain2 = Aura("./encrypted_data", password=password)
    print(f"\nReopened encrypted brain: {brain2.count()} records")
    brain2.close()

    print("\nEncrypted brain demo complete.")


if __name__ == "__main__":
    main()
