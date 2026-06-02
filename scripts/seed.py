"""Seed the chosen vector backend + the structured store from the corpus.

    python scripts/seed.py                      # local: fastembed + LanceDB + SQLite
    RETRIEVAL_BRIDGE_BACKEND=turbopuffer \\
    RETRIEVAL_BRIDGE_EMBEDDER=local \\
        python scripts/seed.py                  # same corpus, turbopuffer backend

The vector store gets the searchable text + flat, filterable attributes. The
structured store gets the per-ticket "account" facts that a retrieved id joins
to (project, plan tier, monthly revenue) — the SQLite analog of the DDN Postgres
model reached via a relationship.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Make `retrieval_bridge` importable when run as `python scripts/seed.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_bridge import RetrievalBridge, StructuredStore, get_backend, get_embedder

CORPUS = Path(__file__).resolve().parent.parent / "data" / "tickets.jsonl"

# Scalar attributes carried into the vector store (filterable at query time).
ATTRS = ["project_id", "plan_tier", "status", "created_at", "component", "error_code", "severity", "root_cause"]


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        print("Corpus not found; generating it…")
        subprocess.run([sys.executable, str(CORPUS.parent / "generate_corpus.py")], check=True)
    with CORPUS.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed retrieval-bridge backends.")
    ap.add_argument("--backend", default=None, help="lancedb | turbopuffer | pgvector")
    ap.add_argument("--embedder", default=None, help="local | openai | voyage | cohere")
    ap.add_argument("--structured-db", default="./.structured.db")
    args = ap.parse_args()

    tickets = load_corpus()
    print(f"Loaded {len(tickets)} tickets from {CORPUS.name}")

    embedder = get_embedder(args.embedder)
    backend = get_backend(args.backend)
    print(f"Embedder: {embedder.name} (dim={embedder.dim})   Backend: {backend.name}")

    # --- vector store ---
    vector_docs = [{"id": t["id"], "text": t["text"], **{a: t[a] for a in ATTRS}} for t in tickets]
    bridge = RetrievalBridge(backend=backend, embedder=embedder)
    print("Embedding + indexing… (first local run downloads the BGE model once)")
    n = bridge.index(vector_docs)
    print(f"Indexed {n} documents into {backend.name}.")

    # --- structured store (the DDN Postgres-join analog) ---
    store = StructuredStore(args.structured_db)
    store.upsert([t["account"] for t in tickets])
    print(f"Wrote {len(tickets)} structured account rows -> {args.structured_db}")
    print("\nSeed complete. Run:  python scripts/demo.py")


if __name__ == "__main__":
    main()
